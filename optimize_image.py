#! /usr/bin/env python

import os
import sys
import argparse
import numpy as np

import settings
from optimize.gradient_optimizer import GradientOptimizer, FindParams
from caffevis.caffevis_helper import check_force_backward_true, read_label_file, load_mean

LR_POLICY_CHOICES = ('constant', 'progress', 'progress01')



def get_parser():
    parser = argparse.ArgumentParser(description='Script to find, with or without regularization, images that cause high or low activations of specific neurons in a network via numerical optimization. Settings are read from settings.py, overridden in settings_local.py, and may be further overridden on the command line.',
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100)
    )

    # Network and data options
    parser.add_argument('--caffe-root', type = str, default = settings.caffevis_caffe_root,
                        help = 'Path to caffe root directory.')
    parser.add_argument('--deploy-proto', type = str, default = settings.caffevis_deploy_prototxt,
                        help = 'Path to caffe network prototxt.')
    parser.add_argument('--net-weights', type = str, default = settings.caffevis_network_weights,
                        help = 'Path to caffe network weights.')
    parser.add_argument('--channel-swap-to-rgb', type = str, default = '(2,1,0)',
                        help = 'Permutation to apply to channels to change to RGB space for plotting. Hint: (0,1,2) if your network is trained for RGB, (2,1,0) if it is trained for BGR.')
    parser.add_argument('--data-size', type = str, default = '(227,227)',
                        help = 'Size of network input.')

    #### FindParams

    # Where to start
    parser.add_argument('--start-at', type = str, default = 'mean_plus_rand', choices = ('mean_plus_rand', 'randu', 'mean'),
                        help = 'How to generate x0, the initial point used in optimization.')
    parser.add_argument('--rand-seed', type = int, default = settings.optimize_image_rand_seed,
                        help = 'Random seed used for generating the start-at image (use different seeds to generate different images).')
    parser.add_argument('--batch-size', type=int, default=settings.optimize_image_batch_size,
                        help = 'Batch size used for generating several images, each index will be used as random seed')

    # What to optimize
    parser.add_argument('--push-layers', type = list, default = settings.optimize_image_push_layers,
                        help = 'Name of layers that contains the desired neuron whose value is optimized.')
    parser.add_argument('--push-channel', type = int, default = '130',
                        help = 'Channel number for desired neuron whose value is optimized (channel for conv, neuron index for FC).')
    parser.add_argument('--push-spatial', type = str, default = 'None',
                        help = 'Which spatial location to push for conv layers. For FC layers, set this to None. For conv layers, set it to a tuple, e.g. when using `--push-layer conv5` on AlexNet, --push-spatial (6,6) will maximize the center unit of the 13x13 spatial grid.')
    parser.add_argument('--push-dir', type = float, default = 1,
                        help = 'Which direction to push the activation of the selected neuron, that is, the value used to begin backprop. For example, use 1 to maximize the selected neuron activation and  -1 to minimize it.')

    # Use regularization?
    parser.add_argument('--decay', type = float, default = settings.optimize_image_decay,
                        help = 'Amount of L2 decay to use.')
    parser.add_argument('--blur-radius', type = float, default = settings.optimize_image_blur_radius,
                        help = 'Radius in pixels of blur to apply after each BLUR_EVERY steps. If 0, perform no blurring. Blur sizes between 0 and 0.3 work poorly.')
    parser.add_argument('--blur-every', type = int, default = settings.optimize_image_blue_every,
                        help = 'Blur every BLUR_EVERY steps. If 0, perform no blurring.')
    parser.add_argument('--small-val-percentile', type = float, default = 0,
                        help = 'Induce sparsity by setting pixels with absolute value under SMALL_VAL_PERCENTILE percentile to 0. Not discussed in paper. 0 to disable.')
    parser.add_argument('--small-norm-percentile', type = float, default = 0,
                        help = 'Induce sparsity by setting pixels with norm under SMALL_NORM_PERCENTILE percentile to 0. \\theta_{n_pct} from the paper. 0 to disable.')
    parser.add_argument('--px-benefit-percentile', type = float, default = 0,
                        help = 'Induce sparsity by setting pixels with contribution under PX_BENEFIT_PERCENTILE percentile to 0. Mentioned briefly in paper but not used. 0 to disable.')
    parser.add_argument('--px-abs-benefit-percentile', type = float, default = 0,
                        help = 'Induce sparsity by setting pixels with contribution under PX_BENEFIT_PERCENTILE percentile to 0. \\theta_{c_pct} from the paper. 0 to disable.')

    # How much to optimize
    parser.add_argument('--lr-policy', type = str, default = settings.optimize_image_lr_policy, choices = LR_POLICY_CHOICES,
                        help = 'Learning rate policy. See description in lr-params.')
    parser.add_argument('--lr-params', type = str, default = settings.optimize_image_lr_params,
                        help = 'Learning rate params, specified as a string that evalutes to a Python dict. Params that must be provided dependon which lr-policy is selected. The "constant" policy requires the "lr" key and uses the constant given learning rate. The "progress" policy requires the "max_lr" and "desired_prog" keys and scales the learning rate such that the objective function will change by an amount equal to DESIRED_PROG under a linear objective assumption, except the LR is limited to MAX_LR. The "progress01" policy requires the "max_lr", "early_prog", and "late_prog_mult" keys and is tuned for optimizing neurons with outputs in the [0,1] range, e.g. neurons on a softmax layer. Under this policy optimization slows down as the output approaches 1 (see code for details).')
    parser.add_argument('--max-iters', type = list, default = settings.optimize_image_max_iters,
                        help = 'List of number of iterations of the optimization loop.')

    # Where to save results
    parser.add_argument('--output-prefix', type = str, default = settings.optimize_image_output_prefix,
                        help = 'Output path and filename prefix (default: outputs/%(p.push_layer)s/unit_%(p.push_channel)04d/opt_%(r.batch_index)03d)')
    parser.add_argument('--brave', action = 'store_true', default=True, help = 'Allow overwriting existing results files. Default: off, i.e. cowardly refuse to overwrite existing files.')
    parser.add_argument('--skipbig', action = 'store_true', default=True, help = 'Skip outputting large *info_big.pkl files (contains pickled version of x0, last x, best x, first x that attained max on the specified layer.')
    parser.add_argument('--skipsmall', action = 'store_true', default=True, help = 'Skip outputting small *info.pkl files (contains pickled version of..')

    return parser



def parse_and_validate_lr_params(parser, lr_policy, lr_params):
    assert lr_policy in LR_POLICY_CHOICES

    try:
        lr_params = eval(lr_params)
    except (SyntaxError,NameError) as _:
        err = 'Tried to parse the following lr_params value\n%s\nas a Python expression, but it failed. lr_params should evaluate to a valid Python dict.' % lr_params
        parser.error(err)

    if lr_policy == 'constant':
        if not 'lr' in lr_params:
            parser.error('Expected lr_params to be dict with at least "lr" key, but dict is %s' % repr(lr_params))
    elif lr_policy == 'progress':
        if not ('max_lr' in lr_params and 'desired_prog' in lr_params):
            parser.error('Expected lr_params to be dict with at least "max_lr" and "desired_prog" keys, but dict is %s' % repr(lr_params))
    elif lr_policy == 'progress01':
        if not ('max_lr' in lr_params and 'early_prog' in lr_params and 'late_prog_mult' in lr_params):
            parser.error('Expected lr_params to be dict with at least "max_lr", "early_prog", and "late_prog_mult" keys, but dict is %s' % repr(lr_params))

    return lr_params



def parse_and_validate_push_spatial(parser, push_spatial):
    '''Returns tuple of length 2.'''
    try:
        push_spatial = eval(push_spatial)
    except (SyntaxError,NameError) as _:
        err = 'Tried to parse the following push_spatial value\n%s\nas a Python expression, but it failed. push_spatial should be a valid Python expression.' % push_spatial
        parser.error(err)

    if push_spatial == None:
        push_spatial = (0,0)    # Convert to tuple format
    elif isinstance(push_spatial, tuple) and len(push_spatial) == 2:
        pass
    else:
        err = 'push_spatial should be None or a valid tuple of indices of length 2, but it is: %s' % push_spatial
        parser.error(err)

    return push_spatial


def get_layer_info(settings, layer_name):
    '''
        get layer info (name, type, input, output, filter, stride, pad) from settings

    :param settings: contains script settings
    :param layer_name: name of layer
    :return: (name, type, input, output, filter, stride, pad)
    '''

    # go over layers
    for (name, type, input, output, filter, stride, pad) in settings.max_tracker_layers_list:
        if name == layer_name:
            return (name, type, input, output, filter, stride, pad)


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Finish parsing args

    lr_params = parse_and_validate_lr_params(parser, args.lr_policy, args.lr_params)
    push_spatial = parse_and_validate_push_spatial(parser, args.push_spatial)

    net_channel_swap = settings.caffe_net_channel_swap

    range_scale = settings.caffe_net_raw_scale
            
    # Load network
    sys.path.insert(0, os.path.join(args.caffe_root, 'python'))
    import caffe
    if settings.caffevis_mode_gpu:
        caffe.set_mode_gpu()
        print 'optimize_image mode (in main thread):     GPU'
    else:
        caffe.set_mode_cpu()
        print 'optimize_image mode (in main thread):     CPU'

    net = caffe.Classifier(
        args.deploy_proto,
        args.net_weights,
        mean=None,  # Set to None for now, assign later         # self._data_mean,
        channel_swap=net_channel_swap,
        raw_scale=range_scale,
        image_dims=settings.caffe_net_image_dims,
    )

    if isinstance(settings.caffevis_data_mean, basestring):
        # If the mean is given as a filename, load the file
        try:
            data_mean = load_mean(settings.caffevis_data_mean)
        except IOError:
            print '\n\nCound not load mean file:', data_mean
            print 'To fetch a default model and mean file, use:\n'
            print '  $ cd models/caffenet-yos/'
            print '  $ cp ./fetch.sh\n\n'
            print 'Or to use your own mean, change caffevis_data_mean in settings_local.py or override by running with `--mean MEAN_FILE` (see --help).\n'
            raise
        input_shape = net.blobs[net.inputs[0]].data.shape[-2:]  # e.g. 227x227

        # Crop center region (e.g. 227x227) if mean is larger (e.g. 256x256)
        excess_h = data_mean.shape[1] - input_shape[0]
        excess_w = data_mean.shape[2] - input_shape[1]
        assert excess_h >= 0 and excess_w >= 0, 'mean should be at least as large as %s' % repr(input_shape)
        data_mean = data_mean[:, (excess_h / 2):(excess_h / 2 + input_shape[0]),
                    (excess_w / 2):(excess_w / 2 + input_shape[1])]
    elif args.data_mean is None:
        pass
    else:
        # The mean has been given as a value or a tuple of values
        data_mean = np.array(settings.caffevis_data_mean)
        # Promote to shape C,1,1
        while len(data_mean.shape) < 3:
            data_mean = np.expand_dims(data_mean, -1)

    print 'Using mean with shape:', data_mean.shape

    if data_mean is not None:
        net.transformer.set_mean(net.inputs[0], data_mean)

    check_force_backward_true(settings.caffevis_deploy_prototxt)

    current_data_shape = net.blobs['data'].shape
    net.blobs['data'].reshape(args.batch_size, current_data_shape[1], current_data_shape[2], current_data_shape[3])
    net.reshape()

    labels = None
    if settings.caffevis_labels:
        labels = read_label_file(settings.caffevis_labels)


    batched_data_mean = np.repeat(data_mean[np.newaxis, :, :, :], args.batch_size, axis=0)
    optimizer = GradientOptimizer(settings, net, batched_data_mean, labels = labels,
                                  label_layers = settings.caffevis_label_layers,
                                  channel_swap_to_rgb = net_channel_swap)

    # go over push layers
    for count, push_layer in enumerate(args.push_layers):

        # get layer type
        (name, type, input, output, filter, stride, pad) = get_layer_info(settings, push_layer)

        if type == 'FullyConnected':
            # get number of units
            channels = output
            push_spatial = (0, 0)
        elif type == 'Convolution':
            # get number of channels
            channels = output[2]
            push_spatial = (filter[0]/2, filter[1]/2)

        # if channels defined in settings file, use them
        if settings.optimize_image_channels:
            channels_list = settings.optimize_image_channels
        else:
            channels_list = range(channels)

        # go over channels
        for current_channel in channels_list:
            params = FindParams(
                start_at = args.start_at,
                rand_seed = args.rand_seed,
                batch_size = args.batch_size,
                push_layer = push_layer,
                push_channel = current_channel,
                push_spatial = push_spatial,
                push_dir = args.push_dir,
                decay = args.decay,
                blur_radius = args.blur_radius,
                blur_every = args.blur_every,
                small_val_percentile = args.small_val_percentile,
                small_norm_percentile = args.small_norm_percentile,
                px_benefit_percentile = args.px_benefit_percentile,
                px_abs_benefit_percentile = args.px_abs_benefit_percentile,
                lr_policy = args.lr_policy,
                lr_params = lr_params,
                max_iter = args.max_iters[count % len(args.max_iters)],
                layer_is_conv=(type == 'Convolution'),
            )

            optimizer.run_optimize(params, prefix_template = args.output_prefix,
                                   brave = args.brave, skipbig = args.skipbig, skipsmall = args.skipsmall)


if __name__ == '__main__':
    main()
