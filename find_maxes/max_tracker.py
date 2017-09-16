#! /usr/bin/env python

import errno
import os
import sys
from datetime import datetime

import numpy as np
from caffe_misc import RegionComputer, save_caffe_image, get_max_data_extent, extract_patch_from_image, \
    compute_data_layer_focus_area
from misc import get_files_from_image_list
from misc import get_files_from_siamese_image_list

from jby_misc import WithTimer
from loaders import load_imagenet_mean, load_labels


def hardcoded_get(settings):
    prototxt = '/home/jyosinsk/results/140311_234854_afadfd3_priv_netbase_upgraded/deploy_1.prototxt'
    weights = '/home/jyosinsk/results/140311_234854_afadfd3_priv_netbase_upgraded/caffe_imagenet_train_iter_450000'
    datadir = '/home/jyosinsk/imagenet2012/val'
    filelist = 'mini_valid.txt'

    imagenet_mean = load_imagenet_mean(settings)
    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    if settings.caffevis_mode_gpu:
        caffe.set_mode_gpu()
        print 'CaffeVisApp mode (in main thread):     GPU'
    else:
        caffe.set_mode_cpu()
        print 'CaffeVisApp mode (in main thread):     CPU'

    net = caffe.Classifier(prototxt, weights,
                           mean=imagenet_mean,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
    net.set_phase_test()
    net.set_mode_cpu()
    labels = load_labels(settings)

    return net, imagenet_mean, labels, datadir, filelist



def mkdir_p(path):
    # From https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



class MaxTracker(object):
    def __init__(self, is_conv, n_channels, n_top = 10, initial_val = -1e99, dtype = 'float32'):
        self.is_conv = is_conv
        self.max_vals = np.ones((n_channels, n_top), dtype = dtype) * initial_val
        self.n_top = n_top
        if is_conv:
            self.max_locs = -np.ones((n_channels, n_top, 5), dtype = 'int')   # image_idx, image_class, selected_input_index, i, j
        else:
            self.max_locs = -np.ones((n_channels, n_top, 3), dtype = 'int')   # image_idx, image_class, selected_input_index

        # set of seen inputs, used to avoid updating on the same input twice
        self.seen_inputs = set()

    def update(self, blob, image_idx, image_class, selected_input_index, layer_unique_input_source):

        if layer_unique_input_source in self.seen_inputs:
            return

        # add input identifier to seen inputs set
        self.seen_inputs.add(layer_unique_input_source)

        data = blob[0]                                        # Note: makes a copy of blob, e.g. (96,55,55)
        n_channels = data.shape[0]
        data_unroll = data.reshape((n_channels, -1))          # Note: no copy eg (96,3025). Does nothing if not is_conv

        maxes = data_unroll.argmax(1)   # maxes for each channel, eg. (96,)

        # if unique_input_source already exist, we can skip the update since we've already seen it


        #insertion_idx = zeros((n_channels,))
        #pdb.set_trace()
        for ii in xrange(n_channels):
            idx = np.searchsorted(self.max_vals[ii], data_unroll[ii, maxes[ii]])
            if idx == 0:
                # Smaller than all 10
                continue
            # Store new max in the proper order. Update both arrays:
            # self.max_vals:
            self.max_vals[ii,:idx-1] = self.max_vals[ii,1:idx]       # shift lower values
            self.max_vals[ii,idx-1] = data_unroll[ii, maxes[ii]]     # store new max value
            # self.max_locs
            self.max_locs[ii,:idx-1] = self.max_locs[ii,1:idx]       # shift lower location data
            # store new location
            if self.is_conv:
                self.max_locs[ii,idx-1] = (image_idx, image_class, selected_input_index) + np.unravel_index(maxes[ii], data.shape[1:])
            else:
                self.max_locs[ii,idx-1] = (image_idx, image_class, selected_input_index)



class NetMaxTracker(object):
    def __init__(self, settings, layers, n_top = 10, initial_val = -1e99, dtype = 'float32'):
        self.layers = layers
        self.init_done = False
        self.n_top = n_top
        self.initial_val = initial_val
        self.settings = settings

    def _init_with_net(self, net):
        self.max_trackers = {}

        for layer in self.layers:

            top_name = net.top_names[layer][0]
            blob = net.blobs[top_name].data

            # normalize layer name, this is used for siamese networks where we want layers "conv_1" and "conv_1_p" to
            # count as the same layer in terms of activations
            normalized_layer_name = self.settings.normalize_layer_name_for_max_tracker_fn(layer)

            is_conv = self.settings.is_conv_fn(layer)

            # only add normalized layer once
            if normalized_layer_name not in self.max_trackers:
                self.max_trackers[normalized_layer_name] = MaxTracker(is_conv, blob.shape[1], n_top = self.n_top,
                                                                      initial_val = self.initial_val,
                                                                      dtype = blob.dtype)

        self.init_done = True

    def update(self, net, image_idx, image_class, net_unique_input_source):
        '''Updates the maxes found so far with the state of the given net. If a new max is found, it is stored together with the image_idx.'''
        if not self.init_done:
            self._init_with_net(net)

        for layer in self.layers:

            top_name = net.top_names[layer][0]
            blob = net.blobs[top_name].data

            normalized_layer_name = self.settings.normalize_layer_name_for_max_tracker_fn(layer)

            # in siamese network, we might need to select one of the images from the siamese pair
            if self.settings.is_siamese:
                selected_input_index = self.settings.siamese_layer_to_index_of_saved_image_fn(layer)

                if selected_input_index == 0:
                    # first image identifier is selected
                    layer_unique_input_source = net_unique_input_source[0]
                elif selected_input_index == 1:
                    # second image identifier is selected
                    layer_unique_input_source = net_unique_input_source[1]
                elif selected_input_index == -1:
                    # both images are selected
                    layer_unique_input_source = net_unique_input_source
            else:
                layer_unique_input_source = net_unique_input_source
                selected_input_index = -1

            self.max_trackers[normalized_layer_name].update(blob, image_idx, image_class, selected_input_index, layer_unique_input_source)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['settings']
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)

        self.settings = None


def load_file_list(settings):

    if (settings.static_files_input_mode == "image_list") and (not settings.is_siamese):
        available_files, labels = get_files_from_image_list(settings)

    elif (settings.static_files_input_mode == "image_list") and (settings.is_siamese):
        available_files, labels = get_files_from_siamese_image_list(settings)

    converted_labels = [settings.convert_label_fn(label) for label in labels]

    return available_files, converted_labels


def scan_images_for_maxes(settings, net, datadir, n_top):
    image_filenames, image_labels = load_file_list(settings)
    print 'Scanning %d files' % len(image_filenames)
    print '  First file', os.path.join(datadir, image_filenames[0])

    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    tracker = NetMaxTracker(settings, n_top = n_top, layers=settings.layers_for_max_tracker)
    for image_idx in xrange(len(image_filenames)):

        filename = image_filenames[image_idx]
        image_class = image_labels[image_idx]
        #im = caffe.io.load_image('../../data/ilsvrc12/mini_ilsvrc_valid/sized/ILSVRC2012_val_00000610.JPEG')
        do_print = (image_idx % 100 == 0)
        if do_print:
            print '%s   Image %d/%d' % (datetime.now().ctime(), image_idx, len(image_filenames))
        with WithTimer('Load image', quiet = not do_print):
            im = caffe.io.load_image(os.path.join(datadir, filename))
        with WithTimer('Predict   ', quiet = not do_print):
            net.predict([im], oversample = False)   # Just take center crop
        with WithTimer('Update    ', quiet = not do_print):
            tracker.update(net, image_idx, image_class, net_unique_input_source=filename)


    print 'done!'
    return tracker


def scan_pairs_for_maxes(settings, net, datadir, n_top):
    image_filenames, image_labels = load_file_list(settings)
    print 'Scanning %d pairs' % len(image_filenames)
    print '  First pair', image_filenames[0]

    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    tracker = NetMaxTracker(settings, n_top=n_top, layers=settings.layers_for_max_tracker)
    for image_idx in xrange(len(image_filenames)):

        images_pair = image_filenames[image_idx]
        filename1 = images_pair[0]
        filename2 = images_pair[1]
        image_class = image_labels[image_idx]
        # im = caffe.io.load_image('../../data/ilsvrc12/mini_ilsvrc_valid/sized/ILSVRC2012_val_00000610.JPEG')
        do_print = (image_idx % 100 == 0)
        if do_print:
            print '%s   Pair %d/%d' % (datetime.now().ctime(), image_idx, len(image_filenames))
        with WithTimer('Load image', quiet=not do_print):
            im1 = caffe.io.load_image(os.path.join(datadir, filename1))
            im2 = caffe.io.load_image(os.path.join(datadir, filename2))

            net_input_dims = net.blobs['data'].data.shape[2:4]
            im1 = caffe.io.resize_image(im1, net_input_dims)
            im2 = caffe.io.resize_image(im2, net_input_dims)

            im = np.concatenate((im1, im2), axis=2)

        with WithTimer('Predict   ', quiet=not do_print):
            net.predict([im], oversample=False)
        with WithTimer('Update    ', quiet=not do_print):
            tracker.update(net, image_idx, image_class, net_unique_input_source=images_pair)

    print 'done!'
    return tracker


def save_representations(settings, net, datadir, filelist, layer, first_N = None):
    image_filenames, image_labels = load_file_list(filelist)
    if first_N is None:
        first_N = len(image_filenames)
    assert first_N <= len(image_filenames)
    image_indices = range(first_N)
    print 'Scanning %d files' % len(image_indices)
    assert len(image_indices) > 0
    print '  First file', os.path.join(datadir, image_filenames[image_indices[0]])

    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    indices = None
    rep = None
    for ii,image_idx in enumerate(image_indices):
        filename = image_filenames[image_idx]
        image_class = image_labels[image_idx]
        do_print = (image_idx % 10 == 0)
        if do_print:
            print '%s   Image %d/%d' % (datetime.now().ctime(), image_idx, len(image_indices))
        with WithTimer('Load image', quiet = not do_print):
            im = caffe.io.load_image(os.path.join(datadir, filename))
        with WithTimer('Predict   ', quiet = not do_print):
            net.predict([im], oversample = False)   # Just take center crop
        with WithTimer('Store     ', quiet = not do_print):
            top_name = net.top_names[layer][0]
            if rep is None:
                rep_shape = net.blobs[top_name].data[0].shape   # e.g. (256,13,13)
                rep = np.zeros((len(image_indices),) + rep_shape)   # e.g. (1000,256,13,13)
                indices = [0] * len(image_indices)
            indices[ii] = image_idx
            rep[ii] = net.blobs[top_name].data[0]

    print 'done!'
    return indices,rep


def output_max_patches(settings, max_tracker, net, layer, idx_begin, idx_end, num_top, datadir, filelist, outdir, do_which):
    do_maxes, do_deconv, do_deconv_norm, do_backprop, do_backprop_norm, do_info = do_which
    assert do_maxes or do_deconv or do_deconv_norm or do_backprop or do_backprop_norm or do_info, 'nothing to do'

    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    mt = max_tracker
    rc = RegionComputer(settings.max_tracker_layers_list)

    image_filenames, image_labels = load_file_list(settings)

    if settings.is_siamese:
        print 'Loaded filenames and labels for %d pairs' % len(image_filenames)
        print '  First pair', image_filenames[0]
    else:
        print 'Loaded filenames and labels for %d files' % len(image_filenames)
        print '  First file', os.path.join(datadir, image_filenames[0])

    num_top_in_mt = mt.max_locs.shape[1]
    assert num_top <= num_top_in_mt, 'Requested %d top images but MaxTracker contains only %d' % (num_top, num_top_in_mt)
    assert idx_end >= idx_begin, 'Range error'

    size_ii, size_jj = get_max_data_extent(net, layer, rc, mt.is_conv)
    data_size_ii, data_size_jj = net.blobs['data'].data.shape[2:4]

    n_total_images = (idx_end-idx_begin) * num_top
    for cc, channel_idx in enumerate(range(idx_begin, idx_end)):
        unit_dir = os.path.join(outdir, layer, 'unit_%04d' % channel_idx)
        mkdir_p(unit_dir)

        if do_info:
            info_filename = os.path.join(unit_dir, 'info.txt')
            info_file = open(info_filename, 'w')
            print >>info_file, '# is_conv val image_idx image_class i(if is_conv) j(if is_conv) filename'

        # iterate through maxes from highest (at end) to lowest
        for max_idx_0 in range(num_top):
            max_idx = num_top_in_mt - 1 - max_idx_0
            if mt.is_conv:
                im_idx, im_class, selected_input_index, ii, jj = mt.max_locs[channel_idx, max_idx]
            else:
                im_idx, im_class, selected_input_index = mt.max_locs[channel_idx, max_idx]
            recorded_val = mt.max_vals[channel_idx, max_idx]
            filename = image_filenames[im_idx]
            do_print = (max_idx_0 == 0)
            if do_print:
                print '%s   Output file/image(s) %d/%d' % (datetime.now().ctime(), cc * num_top, n_total_images)

            [out_ii_start, out_ii_end, out_jj_start, out_jj_end, data_ii_start, data_ii_end, data_jj_start, data_jj_end] = \
                compute_data_layer_focus_area(mt.is_conv, ii, jj, rc, layer, size_ii, size_jj, data_size_ii, data_size_jj)

            if do_info:
                print >>info_file, 1 if mt.is_conv else 0, '%.6f' % mt.max_vals[channel_idx, max_idx],
                if mt.is_conv:
                    print >>info_file, '%d %d %d %d %d' % tuple(mt.max_locs[channel_idx, max_idx]),
                else:
                    print >>info_file, '%d %d %d' % tuple(mt.max_locs[channel_idx, max_idx]),
                print >>info_file, filename

            if not (do_maxes or do_deconv or do_deconv_norm or do_backprop or do_backprop_norm):
                continue

            with WithTimer('Load image', quiet = not do_print):

                if settings.is_siamese:
                    # in siamese network, filename is a pair of image file names
                    filename1 = filename[0]
                    filename2 = filename[1]

                    # load both images
                    im1 = caffe.io.load_image(os.path.join(datadir, filename1))
                    im2 = caffe.io.load_image(os.path.join(datadir, filename2))

                    # resize images according to input dimension
                    net_input_dims = net.blobs['data'].data.shape[2:4]
                    im1 = caffe.io.resize_image(im1, net_input_dims)
                    im2 = caffe.io.resize_image(im2, net_input_dims)

                    # concatenate channelwise
                    im = np.concatenate((im1, im2), axis=2)
                else:
                    im = caffe.io.load_image(os.path.join(datadir, filename))

            with WithTimer('Predict   ', quiet = not do_print):
                net.predict([im], oversample = False)

            # in siamese network, we wish to return from the normalized layer name and selected input index to the
            # denormalized layer name, e.g. from "conv1_1" and selected_input_index=1 to "conv1_1_p"
            denormalized_layer_name = settings.denormalize_layer_name_for_max_tracker_fn(layer, selected_input_index)
            denormalized_top_name = net.top_names[denormalized_layer_name][0]

            if len(net.blobs[denormalized_top_name].data.shape) == 4:
                reproduced_val = net.blobs[denormalized_top_name].data[0,channel_idx,ii,jj]
            else:
                reproduced_val = net.blobs[denormalized_top_name].data[0,channel_idx]
            if abs(reproduced_val - recorded_val) > .1:
                print 'Warning: recorded value %s is suspiciously different from reproduced value %s. Is the filelist the same?' % (recorded_val, reproduced_val)

            if do_maxes:
                #grab image from data layer, not from im (to ensure preprocessing / center crop details match between image and deconv/backprop)

                out_arr = extract_patch_from_image(net.blobs['data'].data[0], net, selected_input_index, settings,
                                                   data_ii_end, data_ii_start, data_jj_end, data_jj_start,
                                                   out_ii_end, out_ii_start, out_jj_end, out_jj_start, size_ii, size_jj)

                with WithTimer('Save img  ', quiet = not do_print):
                    save_caffe_image(out_arr, os.path.join(unit_dir, 'maxim_%03d.png' % max_idx_0),
                                     autoscale = False, autoscale_center = 0)

            if do_deconv or do_deconv_norm:
                diffs = net.blobs[denormalized_top_name].diff * 0
                if len(diffs.shape) == 4:
                    diffs[0,channel_idx,ii,jj] = 1.0
                else:
                    diffs[0,channel_idx] = 1.0
                with WithTimer('Deconv    ', quiet = not do_print):
                    net.deconv_from_layer(denormalized_layer_name, diffs)

                out_arr = extract_patch_from_image(net.blobs['data'].diff[0], net, selected_input_index, settings,
                                                   data_ii_end, data_ii_start, data_jj_end, data_jj_start,
                                                   out_ii_end, out_ii_start, out_jj_end, out_jj_start, size_ii, size_jj)

                if out_arr.max() == 0:
                    print 'Warning: Deconv out_arr in range', out_arr.min(), 'to', out_arr.max(), 'ensure force_backward: true in prototxt'

                if do_deconv:
                    with WithTimer('Save img  ', quiet=not do_print):
                        save_caffe_image(out_arr, os.path.join(unit_dir, 'deconv_%03d.png' % max_idx_0),
                                         autoscale=False, autoscale_center=0)
                if do_deconv_norm:
                    out_arr = np.linalg.norm(out_arr, axis=0)
                    with WithTimer('Save img  ', quiet=not do_print):
                        save_caffe_image(out_arr, os.path.join(unit_dir, 'deconvnorm_%03d.png' % max_idx_0))


            if do_backprop or do_backprop_norm:
                diffs = net.blobs[denormalized_top_name].diff * 0

                if len(diffs.shape) == 4:
                    diffs[0,channel_idx,ii,jj] = 1.0
                else:
                    diffs[0,channel_idx] = 1.0

                with WithTimer('Backward  ', quiet = not do_print):
                    net.backward_from_layer(denormalized_layer_name, diffs)

                out_arr = extract_patch_from_image(net.blobs['data'].diff[0], net, selected_input_index, settings,
                                                   data_ii_end, data_ii_start, data_jj_end, data_jj_start,
                                                   out_ii_end, out_ii_start, out_jj_end, out_jj_start, size_ii, size_jj)

                if out_arr.max() == 0:
                    print 'Warning: Deconv out_arr in range', out_arr.min(), 'to', out_arr.max(), 'ensure force_backward: true in prototxt'
                if do_backprop:
                    with WithTimer('Save img  ', quiet = not do_print):
                        save_caffe_image(out_arr, os.path.join(unit_dir, 'backprop_%03d.png' % max_idx_0),
                                         autoscale = False, autoscale_center = 0)
                if do_backprop_norm:
                    out_arr = np.linalg.norm(out_arr, axis=0)
                    with WithTimer('Save img  ', quiet = not do_print):
                        save_caffe_image(out_arr, os.path.join(unit_dir, 'backpropnorm_%03d.png' % max_idx_0))

        if do_info:
            info_file.close()

