#! /usr/bin/env python

import os
import errno
import pickle
import datetime
import StringIO
from pylab import *
from scipy.ndimage.filters import gaussian_filter

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from misc import mkdir_p, combine_dicts
from image_misc import saveimagesc, saveimagescc

from caffe_misc import RegionComputer, get_max_data_extent, compute_data_layer_focus_area, extract_patch_from_image, \
    layer_name_to_top_name


class FindParams(object):
    def __init__(self, **kwargs):
        default_params = dict(
            # Starting
            rand_seed = 0,
            start_at = 'mean_plus_rand',
            batch_size = 9,
            
            # Optimization
            push_layer = 'prob',
            push_channel = 278,
            push_spatial = (0,0),
            push_dir = 1.0,
            decay = .01,
            blur_radius = None,   # 0 or at least .3
            blur_every = 0,       # 0 to skip blurring
            small_val_percentile = None,
            small_norm_percentile = None,
            px_benefit_percentile = None,
            px_abs_benefit_percentile = None,
            is_spatial = False,

            lr_policy = 'constant',
            lr_params = {'lr': 10.0},

            # Terminating
            max_iter = 300)

        self.__dict__.update(default_params)

        for key,val in kwargs.iteritems():
            assert key in self.__dict__, 'Unknown param: %s' % key
            self.__dict__[key] = val

        self._validate_and_normalize()

    def _validate_and_normalize(self):
        if self.lr_policy == 'progress01':
            assert 'max_lr' in self.lr_params
            assert 'early_prog' in self.lr_params
            assert 'late_prog_mult' in self.lr_params
        elif self.lr_policy == 'progress':
            assert 'max_lr' in self.lr_params
            assert 'desired_prog' in self.lr_params
        elif self.lr_policy == 'constant':
            assert 'lr' in self.lr_params
        else:
            raise Exception('Unknown lr_policy: %s' % self.lr_policy)

        assert isinstance(self.push_channel, int), 'push_channel should be an int'
        assert isinstance(self.push_spatial, tuple) and len(self.push_spatial) == 2, 'push_spatial should be a length 2 tuple'    

        # Concatenate push_channel and push_spatial into push_unit and add to params for conveninece
        self.push_unit = (self.push_channel,) + self.push_spatial

    def __str__(self):
        ret = StringIO.StringIO()
        print >>ret, 'FindParams:'
        for key in sorted(self.__dict__.keys()):
            print >>ret, '%30s: %s' % (key, self.__dict__[key])
        return ret.getvalue()



class FindResults(object):
    def __init__(self,batch_index):
        self.ii = []
        self.obj = []
        self.idxmax = []
        self.ismax = []
        self.norm = []
        self.dist = []
        self.std = []
        self.x0 = None
        self.majority_obj = None
        self.majority_xx = None
        self.best_obj = None
        self.best_xx = None
        self.last_obj = None
        self.last_xx = None
        self.meta_result = None
        self.batch_index = batch_index
        
    def update(self, params, ii, acts, idxmax, xx, x0):
        assert params.push_dir > 0, 'push_dir < 0 not yet supported'
        
        self.ii.append(ii)
        self.obj.append(acts[params.push_unit])
        self.idxmax.append(idxmax)
        self.ismax.append(idxmax == params.push_unit)
        self.norm.append(norm(xx))
        self.dist.append(norm(xx-x0))
        self.std.append(xx.flatten().std())
        if self.x0 is None:
            self.x0 = x0.copy()

        # Snapshot when the unit first becomes the highest of its layer
        if params.push_unit == idxmax and self.majority_xx is None:
            self.majority_obj = self.obj[-1]
            self.majority_xx = xx.copy()
            self.majority_ii = ii

        # Snapshot of best-ever objective
        if self.obj[-1] > self.best_obj:
            self.best_obj = self.obj[-1]
            self.best_xx = xx.copy()
            self.best_ii = ii

        # Snapshot of last
        self.last_obj = self.obj[-1]
        self.last_xx = xx.copy()
        self.last_ii = ii

    def trim_arrays(self):
        '''Destructively drop arrays and replace with strings
        containing first couple values; useful for saving results as a
        reasonably sized pickle file.
        '''
        for key,val in self.__dict__.iteritems():
            if isinstance(val, ndarray):
                valstr = '%s array [%s, %s, ...]' % (val.shape, val.flatten()[0], val.flatten()[1])
                self.__dict__[key] = 'Trimmed %s' % valstr

    def __str__(self):
        ret = StringIO.StringIO()
        print >>ret, 'FindResults[%d]:' % self.batch_index
        for key in sorted(self.__dict__.keys()):
            val = self.__dict__[key]
            if isinstance(val, list) and len(val) > 4:
                valstr = '[%s, %s, ..., %s, %s]' % (val[0], val[1], val[-2], val[-1])
            elif isinstance(val, ndarray):
                valstr = '%s array [%s, %s, ...]' % (val.shape, val.flatten()[0], val.flatten()[1])
            else:
                valstr = '%s' % val
            print >>ret, '%30s: %s' % (key, valstr)
        return ret.getvalue()



class GradientOptimizer(object):
    '''Finds images by gradient.'''
    
    def __init__(self, settings, net, batched_data_mean, labels = None, label_layers = [], channel_swap_to_rgb = None):
        self.settings = settings
        self.net = net
        self.batched_data_mean = batched_data_mean
        self.labels = labels if labels else ['labels not provided' for ii in range(1000)]
        self.label_layers = label_layers if label_layers else list()
        if channel_swap_to_rgb:
            self.channel_swap_to_rgb = array(channel_swap_to_rgb)
        else:

            if settings._calculated_is_gray_model:
                self.channel_swap_to_rgb = arange(1)
            else:
                self.channel_swap_to_rgb = arange(3)  # Don't change order

        # since we have a batch of same data mean images, we can just take the first
        if batched_data_mean is not None:
            self._data_mean_rgb_img = self.batched_data_mean[0, self.channel_swap_to_rgb].transpose((1,2,0))  # Store as (227,227,3) in RGB order.
        else:
            self._data_mean_rgb_img = None

    def run_optimize(self, params, prefix_template = None, brave = False, skipbig = False, skipsmall = False):
        '''All images are in Caffe format, e.g. shape (3, 227, 227) in BGR order.'''

        print '\n\nStarting optimization with the following parameters:'
        print params
        
        x0 = self._get_x0(params)
        xx, results, results_generated = self._optimize(params, x0, prefix_template)
        if results_generated:
            self.save_results(params, results, prefix_template, brave = brave, skipbig = skipbig, skipsmall = skipsmall)
            print str([results[batch_index].meta_result for batch_index in range(params.batch_size)])
        
        return xx

    def _get_x0(self, params):
        '''Chooses a starting location'''

        np.random.seed(params.rand_seed)

        input_shape = self.net.blobs['data'].data.shape

        if params.start_at == 'mean_plus_rand':
            x0 = np.random.normal(0, 10, input_shape)
        elif params.start_at == 'randu':
            if self.batched_data_mean is not None:
                x0 = uniform(0, 255, input_shape) - self.batched_data_mean
            else:
                x0 = uniform(0, 255, input_shape)
        elif params.start_at == 'mean':
            x0 = zeros(input_shape)
        else:
            raise Exception('Unknown start conditions: %s' % params.start_at)

        return x0
        
    def _optimize(self, params, x0, prefix_template):
        xx = x0.copy()

        results = [FindResults(batch_index) for batch_index in range(params.batch_size)]

        # check if all required outputs exist, in which case skip this optimization
        all_outputs = [self.generate_output_names(batch_index, params, results, prefix_template) for batch_index in range(params.batch_size)]
        relevant_outputs = [best_X_name for [best_X_name, best_Xpm_name, majority_X_name, majority_Xpm_name, info_name, info_pkl_name, info_big_pkl_name] in all_outputs]
        relevant_outputs_exist = [os.path.exists(best_X_name) for best_X_name in relevant_outputs]
        if all(relevant_outputs_exist):
            return xx, results, False

        # Whether or not the unit being optimized corresponds to a label (e.g. one of the 1000 imagenet classes)
        is_labeled_unit = params.push_layer in self.label_layers

        # Sanity checks for conv vs FC layers
        top_name = layer_name_to_top_name(self.net, params.push_layer)
        data_shape = self.net.blobs[top_name].data.shape
        assert len(data_shape) in (2,4), 'Expected shape of length 2 (for FC) or 4 (for conv) layers but shape is %s' % repr(data_shape)
        is_spatial = (len(data_shape) == 4)

        if is_spatial:
            if params.push_spatial == (0,0):
                recommended_spatial = (data_shape[2]/2, data_shape[3]/2)
                print ('WARNING: A unit on a conv layer (%s) is being optimized, but push_spatial\n'
                       'is %s, so the upper-left unit in the channel is selected. To avoid edge\n'
                       'effects, you might want to optimize a non-edge unit instead, e.g. the center\n'
                       'unit by using `--push_spatial "%s"`\n'
                       % (params.push_layer, params.push_spatial, recommended_spatial))
        else:
            assert params.push_spatial == (0,0), 'For FC layers, spatial indices must be (0,0)'
        
        if is_labeled_unit:
            # Sanity check
            push_label = self.labels[params.push_unit[0]]
        else:
            push_label = None

        old_obj = np.zeros(params.batch_size)
        obj = np.zeros(params.batch_size)
        for ii in range(params.max_iter):
            # 0. Crop data
            if self.batched_data_mean is not None:
                xx = minimum(255.0, maximum(0.0, xx + self.batched_data_mean)) - self.batched_data_mean     # Crop all values to [0,255]
            else:
                xx = minimum(255.0, maximum(0.0, xx)) # Crop all values to [0,255]
            # 1. Push data through net
            out = self.net.forward_all(data = xx)
            #shownet(net)
            top_name = layer_name_to_top_name(self.net, params.push_layer)
            acts = self.net.blobs[top_name].data

            # note: no batch support in 'siamese_batch_pair'
            if self.settings.is_siamese and self.settings.siamese_network_format == 'siamese_batch_pair' and acts.shape[0] == 2:

                if not is_spatial:
                    # promote to 4D
                    acts = np.reshape(acts, (2, -1, 1, 1))
                reshaped_acts = np.reshape(acts, (2, -1))
                idxmax = unravel_index(reshaped_acts.argmax(axis=1), acts.shape[1:])
                valmax = reshaped_acts.max(axis=1)

                # idxmax for fc or prob layer will be like:  (batch,278, 0, 0)
                # idxmax for conv layer will be like:        (batch,37, 4, 37)
                obj[0] = acts[0, params.push_unit[0], params.push_unit[1], params.push_unit[2]]

            elif self.settings.is_siamese and self.settings.siamese_network_format == 'siamese_batch_pair' and acts.shape[0] == 1:

                if not is_spatial:
                    # promote to 4D
                    acts = np.reshape(acts, (1, -1, 1, 1))
                reshaped_acts = np.reshape(acts, (1, -1))
                idxmax = unravel_index(reshaped_acts.argmax(axis=1), acts.shape[1:])
                valmax = reshaped_acts.max(axis=1)

                # idxmax for fc or prob layer will be like:  (batch,278, 0, 0)
                # idxmax for conv layer will be like:        (batch,37, 4, 37)
                obj[0] = acts[0, params.push_unit[0], params.push_unit[1], params.push_unit[2]]

            else:
                if not is_spatial:
                    # promote to 4D
                    acts = np.reshape(acts, (params.batch_size, -1, 1, 1))
                reshaped_acts = np.reshape(acts, (params.batch_size, -1))
                idxmax = unravel_index(reshaped_acts.argmax(axis=1), acts.shape[1:])
                valmax = reshaped_acts.max(axis=1)

                # idxmax for fc or prob layer will be like:  (batch,278, 0, 0)
                # idxmax for conv layer will be like:        (batch,37, 4, 37)
                obj = acts[np.arange(params.batch_size), params.push_unit[0], params.push_unit[1], params.push_unit[2]]

            # 2. Update results
            for batch_index in range(params.batch_size):
                results[batch_index].update(params, ii, acts[batch_index], \
                                            (idxmax[0][batch_index],idxmax[1][batch_index],idxmax[2][batch_index]), \
                                            xx[batch_index], x0[batch_index])

                # 3. Print progress
                if ii > 0:
                    if params.lr_policy == 'progress':
                        print 'iter %-4d batch_index %d progress predicted: %g, actual: %g' % (ii, batch_index, pred_prog[batch_index], obj[batch_index] - old_obj[batch_index])
                    else:
                        print 'iter %-4d batch_index %d progress: %g' % (ii, batch_index, obj[batch_index] - old_obj[batch_index])
                else:
                    print 'iter %d batch_index %d' % (ii, batch_index)
                old_obj[batch_index] = obj[batch_index]

                push_label_str = ('(%s)' % push_label) if is_labeled_unit else ''
                max_label_str  = ('(%s)' % self.labels[idxmax[0][batch_index]]) if is_labeled_unit else ''
                print '     push unit: %16s with value %g %s' % (params.push_unit, acts[batch_index][params.push_unit], push_label_str)
                print '       Max idx: %16s with value %g %s' % ((idxmax[0][batch_index],idxmax[1][batch_index],idxmax[2][batch_index]), valmax[batch_index], max_label_str)
                print '             X:', xx[batch_index].min(), xx[batch_index].max(), norm(xx[batch_index])


            # 4. Do backward pass to get gradient
            top_name = layer_name_to_top_name(self.net, params.push_layer)
            diffs = self.net.blobs[top_name].diff * 0
            if not is_spatial:
                # Promote bc -> bc01
                diffs = diffs[:,:,np.newaxis,np.newaxis]

            if self.settings.is_siamese and self.settings.siamese_network_format == 'siamese_batch_pair' and acts.shape[0] == 2:
                diffs[0, params.push_unit[0], params.push_unit[1], params.push_unit[2]] = params.push_dir
            elif self.settings.is_siamese and self.settings.siamese_network_format == 'siamese_batch_pair' and acts.shape[0] == 1:
                diffs[0, params.push_unit[0], params.push_unit[1], params.push_unit[2]] = params.push_dir
            else:
                diffs[np.arange(params.batch_size), params.push_unit[0], params.push_unit[1], params.push_unit[2]] = params.push_dir
            backout = self.net.backward_from_layer(params.push_layer, diffs if is_spatial else diffs[:,:,0,0])

            grad = backout['data'].copy()
            reshaped_grad = np.reshape(grad, (params.batch_size, -1))
            norm_grad = np.linalg.norm(reshaped_grad, axis=1)
            min_grad = np.amin(reshaped_grad, axis=1)
            max_grad = np.amax(reshaped_grad, axis=1)

            for batch_index in range(params.batch_size):
                print ' layer: %s, channel: %d, batch_index: %d    min grad: %f, max grad: %f, norm grad: %f' % (params.push_layer, params.push_unit[0], batch_index, min_grad[batch_index], max_grad[batch_index], norm_grad[batch_index])
                if norm_grad[batch_index] == 0:
                    print ' batch_index: %d, Grad exactly 0, failed' % batch_index
                    results[batch_index].meta_result = 'Metaresult: grad 0 failure'
                    break

            # 5. Pick gradient update per learning policy
            if params.lr_policy == 'progress01':
                # Useful for softmax layer optimization, taper off near 1
                late_prog = params.lr_params['late_prog_mult'] * (1-obj)
                desired_prog = np.amin(np.stack((np.repeat(params.lr_params['early_prog'], params.batch_size), late_prog), axis=1), axis=1)
                prog_lr = desired_prog / np.square(norm_grad)
                lr = np.amin(np.stack((np.repeat(params.lr_params['max_lr'], params.batch_size), prog_lr), axis=1), axis=1)
                print '    entire batch, desired progress:', desired_prog, 'prog_lr:', prog_lr, 'lr:', lr
                pred_prog = lr * np.sum(np.abs(reshaped_grad) ** 2, axis=-1)
            elif params.lr_policy == 'progress':
                # straight progress-based lr
                prog_lr = params.lr_params['desired_prog'] / (norm_grad**2)
                lr = np.amin(np.stack((np.repeat(params.lr_params['max_lr'], params.batch_size), prog_lr), axis=1), axis=1)
                print '    entire batch, desired progress:', params.lr_params['desired_prog'], 'prog_lr:', prog_lr, 'lr:', lr
                pred_prog = lr * np.sum(np.abs(reshaped_grad) ** 2, axis=-1)
            elif params.lr_policy == 'constant':
                # constant fixed learning rate
                lr = np.repeat(params.lr_params['lr'], params.batch_size)
            else:
                raise Exception('Unimplemented lr_policy')

            for batch_index in range(params.batch_size):

                # 6. Apply gradient update and regularizations
                if ii < params.max_iter-1:
                    # Skip gradient and regularizations on the very last step (so the above printed info is valid for the last step)
                    xx[batch_index] += lr[batch_index] * grad[batch_index]
                    xx[batch_index] *= (1 - params.decay)

                    channels = xx.shape[1]

                    if params.blur_every is not 0 and params.blur_radius > 0:
                        if params.blur_radius < .3:
                            print 'Warning: blur-radius of .3 or less works very poorly'
                            #raise Exception('blur-radius of .3 or less works very poorly')
                        if ii % params.blur_every == 0:
                            for channel in range(channels):
                                cimg = gaussian_filter(xx[batch_index,channel], params.blur_radius)
                                xx[batch_index,channel] = cimg
                    if params.small_val_percentile > 0:
                        small_entries = (abs(xx[batch_index]) < percentile(abs(xx[batch_index]), params.small_val_percentile))
                        xx[batch_index] = xx[batch_index] - xx[batch_index]*small_entries   # e.g. set smallest 50% of xx to zero

                    if params.small_norm_percentile > 0:
                        pxnorms = norm(xx[batch_index,np.newaxis,:,:,:], axis=1)
                        smallpx = pxnorms < percentile(pxnorms, params.small_norm_percentile)
                        smallpx3 = tile(smallpx[:,newaxis,:,:], (1,channels,1,1))
                        xx[batch_index,:,:,:] = xx[batch_index,np.newaxis,:,:,:] - xx[batch_index,np.newaxis,:,:,:]*smallpx3

                    if params.px_benefit_percentile > 0:
                        pred_0_benefit = grad[batch_index,np.newaxis,:,:,:] * -xx[batch_index,np.newaxis,:,:,:]
                        px_benefit = pred_0_benefit.sum(1)   # sum over color channels
                        smallben = px_benefit < percentile(px_benefit, params.px_benefit_percentile)
                        smallben3 = tile(smallben[:,newaxis,:,:], (1,channels,1,1))
                        xx[batch_index,:,:,:] = xx[batch_index,np.newaxis,:,:,:] - xx[batch_index,np.newaxis,:,:,:]*smallben3

                    if params.px_abs_benefit_percentile > 0:
                        pred_0_benefit = grad[batch_index,np.newaxis,:,:,:] * -xx[batch_index,np.newaxis,:,:,:]
                        px_benefit = pred_0_benefit.sum(1)   # sum over color channels
                        smallaben = abs(px_benefit) < percentile(abs(px_benefit), params.px_abs_benefit_percentile)
                        smallaben3 = tile(smallaben[:,newaxis,:,:], (1,channels,1,1))
                        xx[batch_index,:,:,:] = xx[batch_index,np.newaxis,:,:,:] - xx[batch_index,np.newaxis,:,:,:]*smallaben3

            print '     timestamp:', datetime.datetime.now()

        for batch_index in range(params.batch_size):
            if results[batch_index].meta_result is None:
                if results[batch_index].majority_obj is not None:
                    results[batch_index].meta_result = 'batch_index: %d, Metaresult: majority success' % batch_index
                else:
                    results[batch_index].meta_result = 'batch_index: %d, Metaresult: majority failure' % batch_index

        return xx, results, True

    def find_selected_input_index(self, layer_name):

        for item in self.settings.layers_list:

            # if we have only a single layer, the header is the layer name
            if item['format'] == 'normal' and item['name/s'] == layer_name:
                return -1

            # if we got a pair of layers
            elif item['format'] == 'siamese_layer_pair':

                if item['name/s'][0] == layer_name:
                    return 0

                if item['name/s'][1] == layer_name:
                    return 1

            elif item['format'] == 'siamese_batch_pair' and item['name/s'] == layer_name:
                return 0

        return -1

    def generate_output_names(self, batch_index, params, results, prefix_template):

        results_and_params = combine_dicts((('p.', params.__dict__),
                                            ('r.', results[batch_index].__dict__)))
        prefix = prefix_template % results_and_params

        if os.path.isdir(prefix):
            if prefix[-1] != '/':
                prefix += '/'  # append slash for dir-only template
        else:
            dirname = os.path.dirname(prefix)
            if dirname:
                mkdir_p(dirname)

        best_X_name = '%s_best_X.jpg' % prefix
        best_Xpm_name = '%s_best_Xpm.jpg' % prefix
        majority_X_name = '%s_majority_X.jpg' % prefix
        majority_Xpm_name = '%s_majority_Xpm.jpg' % prefix
        info_name = '%s_info.txt' % prefix
        info_pkl_name = '%s_info.pkl' % prefix
        info_big_pkl_name = '%s_info_big.pkl' % prefix
        return [best_X_name, best_Xpm_name, majority_X_name, majority_Xpm_name, info_name, info_pkl_name, info_big_pkl_name]

    def save_results(self, params, results, prefix_template, brave = False, skipbig = False, skipsmall = False):
        if prefix_template is None:
            return

        for batch_index in range(params.batch_size):

            [best_X_name, best_Xpm_name, majority_X_name, majority_Xpm_name, info_name, info_pkl_name, info_big_pkl_name] = \
                self.generate_output_names(batch_index, params, results, prefix_template)

            # Don't overwrite previous results
            if os.path.exists(info_name) and not brave:
                raise Exception('Cowardly refusing to overwrite ' + info_name)

            output_majority = False
            if output_majority:
                # NOTE: this section wasn't tested after changes to code, so some minor index tweaking are in order
                if results[batch_index].majority_xx is not None:
                    asimg = results[batch_index].majority_xx[self.channel_swap_to_rgb].transpose((1,2,0))
                    saveimagescc(majority_X_name, asimg, 0)
                    saveimagesc(majority_Xpm_name, asimg + self._data_mean_rgb_img)  # PlusMean

            if results[batch_index].best_xx is not None:
                # results[batch_index].best_xx.shape is (6,224,224)

                def save_output(data, channel_swap_to_rgb, best_X_image_name):
                                # , best_Xpm_image_name, data_mean_rgb_img):
                    asimg = data[channel_swap_to_rgb].transpose((1, 2, 0))
                    saveimagescc(best_X_image_name, asimg, 0)

                # get center position, relative to layer, of best maximum
                [temp_ii, temp_jj] = results[batch_index].idxmax[results[batch_index].best_ii][1:3]

                is_spatial = params.is_spatial
                layer_name = params.push_layer
                size_ii, size_jj = get_max_data_extent(self.net, self.settings, layer_name, is_spatial)
                data_size_ii, data_size_jj = self.net.blobs['data'].data.shape[2:4]

                [out_ii_start, out_ii_end, out_jj_start, out_jj_end, data_ii_start, data_ii_end, data_jj_start, data_jj_end] = \
                    compute_data_layer_focus_area(is_spatial, temp_ii, temp_jj, self.settings, layer_name, size_ii, size_jj, data_size_ii, data_size_jj)

                selected_input_index = self.find_selected_input_index(layer_name)

                out_arr = extract_patch_from_image(results[batch_index].best_xx, self.net, selected_input_index, self.settings,
                                                   data_ii_end, data_ii_start, data_jj_end, data_jj_start,
                                                   out_ii_end, out_ii_start, out_jj_end, out_jj_start, size_ii, size_jj)

                if self.settings.is_siamese:
                    save_output(out_arr,
                                channel_swap_to_rgb=self.channel_swap_to_rgb[[0, 1, 2]],
                                best_X_image_name=best_X_name)
                else:
                    save_output(out_arr,
                                channel_swap_to_rgb=self.channel_swap_to_rgb,
                                best_X_image_name=best_X_name)

                if self.settings.optimize_image_generate_plus_mean:
                    out_arr_pm = extract_patch_from_image(results[batch_index].best_xx + self.batched_data_mean, self.net, selected_input_index, self.settings,
                                                       data_ii_end, data_ii_start, data_jj_end, data_jj_start,
                                                       out_ii_end, out_ii_start, out_jj_end, out_jj_start, size_ii, size_jj)

                    if self.settings.is_siamese:
                        save_output(out_arr_pm,
                                    channel_swap_to_rgb=self.channel_swap_to_rgb[[0, 1, 2]],
                                    best_X_image_name=best_Xpm_name)
                    else:
                        save_output(out_arr_pm,
                                    channel_swap_to_rgb=self.channel_swap_to_rgb,
                                    best_X_image_name=best_Xpm_name)

            with open(info_name, 'w') as ff:
                print >>ff, params
                print >>ff
                print >>ff, results[batch_index]
            if not skipbig:
                with open(info_big_pkl_name, 'w') as ff:
                    pickle.dump((params, results[batch_index]), ff, protocol=-1)
            if not skipsmall:
                results[batch_index].trim_arrays()
                with open(info_pkl_name, 'w') as ff:
                    pickle.dump((params, results[batch_index]), ff, protocol=-1)

