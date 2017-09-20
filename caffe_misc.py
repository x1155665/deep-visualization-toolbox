#! /usr/bin/env python

import skimage.io
import numpy as np
from image_misc import norm01c


def shownet(net):
    '''Print some stats about a net and its activations'''
    
    print '%-41s%-31s%s' % ('', 'acts', 'act diffs')
    print '%-45s%-31s%s' % ('', 'params', 'param diffs')
    for k, v in net.blobs.items():
        if k in net.params:
            params = net.params[k]
            for pp, blob in enumerate(params):
                if pp == 0:
                    print '  ', 'P: %-5s'%k,
                else:
                    print ' ' * 11,
                print '%-32s' % repr(blob.data.shape),
                print '%-30s' % ('(%g, %g)' % (blob.data.min(), blob.data.max())),
                print '(%g, %g)' % (blob.diff.min(), blob.diff.max())
        print '%-5s'%k, '%-34s' % repr(v.data.shape),
        print '%-30s' % ('(%g, %g)' % (v.data.min(), v.data.max())),
        print '(%g, %g)' % (v.diff.min(), v.diff.max())



def region_converter(top_slice, bot_size, top_size, filter_width = (1,1), stride = (1,1), pad = (0,0), crop_to_boundary = True):
    '''
    Works for conv or pool

vector<int> ConvolutionLayer<Dtype>::JBY_region_of_influence(const vector<int>& slice) {
    +  CHECK_EQ(slice.size(), 4) << "slice must have length 4 (ii_start, ii_end, jj_start, jj_end)";
    +  // Crop region to output size
    +  vector<int> sl = vector<int>(slice);
    +  sl[0] = max(0, min(height_out_, slice[0]));
    +  sl[1] = max(0, min(height_out_, slice[1]));
    +  sl[2] = max(0, min(width_out_, slice[2]));
    +  sl[3] = max(0, min(width_out_, slice[3]));
    +  vector<int> roi;
    +  roi.resize(4);
    +  roi[0] = sl[0] * stride_h_ - pad_h_;
    +  roi[1] = (sl[1]-1) * stride_h_ + kernel_h_ - pad_h_;
    +  roi[2] = sl[2] * stride_w_ - pad_w_;
    +  roi[3] = (sl[3]-1) * stride_w_ + kernel_w_ - pad_w_;
    +  return roi;
    +}
    '''
    assert len(top_slice) == 4
    assert len(bot_size) == 2
    assert len(top_size) == 2
    assert len(filter_width) == 2
    assert len(stride) == 2
    assert len(pad) == 2

    # Crop top slice to allowable region
    top_slice = [ss for ss in top_slice]   # Copy list or array -> list

    # should we crop to boundary
    if crop_to_boundary:
        top_slice[0] = max(0, min(top_size[0], top_slice[0]))
        top_slice[1] = max(0, min(top_size[0], top_slice[1]))
        top_slice[2] = max(0, min(top_size[1], top_slice[2]))
        top_slice[3] = max(0, min(top_size[1], top_slice[3]))

    bot_slice = [-123] * 4

    bot_slice[0] = top_slice[0] * stride[0] - pad[0]
    bot_slice[1] = top_slice[1] * stride[0] - pad[0] + filter_width[0] - 1
    bot_slice[2] = top_slice[2] * stride[1] - pad[1]
    bot_slice[3] = top_slice[3] * stride[1] - pad[1] + filter_width[1] - 1

    return bot_slice


def get_conv_converter(bot_size, top_size, filter_width = (1,1), stride = (1,1), pad = (0,0)):
    return lambda top_slice, crop_to_boundary : region_converter(top_slice, bot_size, top_size, filter_width, stride, pad, crop_to_boundary)


def get_pool_converter(bot_size, top_size, filter_width = (1,1), stride = (1,1), pad = (0,0)):
    return lambda top_slice, crop_to_boundary : region_converter(top_slice, bot_size, top_size, filter_width, stride, pad, crop_to_boundary)



class RegionComputer(object):
    '''Computes regions of possible influcence from higher layers to lower layers.'''

    def __init__(self, layers_list):
        #self.net = net

        _tmp = []

        for (name, type, input, output, filter, stride, pad) in layers_list:
            if type == 'Input':
                _tmp.append((name, None))
            elif type == 'Convolution':
                _tmp.append((name, get_conv_converter(input[0:2], output[0:2], filter, stride, pad)))
            elif type == 'Pooling':
                _tmp.append((name, get_pool_converter(input[0:2], output[0:2], filter, stride, pad)))
            else:
                continue # skip adding layer

        self.names = [tt[0] for tt in _tmp]
        self.converters = [tt[1] for tt in _tmp]

    def convert_region(self, from_layer, to_layer, region, verbose = False, crop_to_boundary = True):
        '''region is the slice of the from_layer in the following Python
            index format: (ii_start, ii_end, jj_start, jj_end)
        '''

        from_idx = self.names.index(from_layer)
        to_idx = self.names.index(to_layer)
        assert from_idx >= to_idx, 'wrong order of from_layer and to_layer'

        ret = region
        for ii in range(from_idx, to_idx, -1):
            converter = self.converters[ii]
            if verbose:
                print 'pushing', self.names[ii], 'region', ret, 'through converter'
            ret = converter(ret, crop_to_boundary)
        if verbose:
            print 'Final region at ', self.names[to_idx], 'is', ret

        return ret


def save_caffe_image(img, filename, autoscale = True, autoscale_center = None):
    '''Takes an image in caffe format (01) or (c01, BGR) and saves it to a file'''
    if len(img.shape) == 2:
        # upsample grayscale 01 -> 01c
        img = np.tile(img[:,:,np.newaxis], (1,1,3))
    else:
        img = img[::-1].transpose((1,2,0))
    if autoscale_center is not None:
        img = norm01c(img, autoscale_center)
    elif autoscale:
        img = img.copy()
        img -= img.min()
        img *= 1.0 / (img.max() + 1e-10)
    skimage.io.imsave(filename, img)


def layer_name_to_top_name(net, layer_name):
    return net.top_names[layer_name][0]


def get_max_data_extent(net, layer_name, rc, is_conv):
    '''Gets the maximum size of the data layer that can influence a unit on layer.'''

    data_size = net.blobs['data'].data.shape[2:4]  # e.g. (227,227) for fc6,fc7,fc8,prop

    if is_conv:
        top_name = layer_name_to_top_name(net, layer_name)
        conv_size = net.blobs[top_name].data.shape[2:4]        # e.g. (13,13) for conv5
        layer_slice_middle = (conv_size[0]/2,conv_size[0]/2+1, conv_size[1]/2,conv_size[1]/2+1)   # e.g. (6,7,6,7,), the single center unit
        data_slice = rc.convert_region(layer_name, 'data', layer_slice_middle, crop_to_boundary = False)
        data_slice_size = data_slice[1]-data_slice[0], data_slice[3]-data_slice[2]   # e.g. (163, 163) for conv5
        # crop data slice size to data size
        data_slice_size = min(data_slice_size[0], data_size[0]), min(data_slice_size[1], data_size[1])
        return data_slice_size
    else:
        # Whole data region
        return data_size


def compute_data_layer_focus_area(is_conv, ii, jj, region_computer, layer_name, size_ii, size_jj, data_size_ii, data_size_jj):

    if is_conv:

        # Compute the focus area of the data layer
        layer_indices = (ii, ii + 1, jj, jj + 1)
        data_indices = region_computer.convert_region(layer_name, 'data', layer_indices)
        data_ii_start, data_ii_end, data_jj_start, data_jj_end = data_indices

        # safe guard edges
        data_ii_start = max(data_ii_start, 0)
        data_jj_start = max(data_jj_start, 0)
        data_ii_end = min(data_ii_end, data_size_ii)
        data_jj_end = min(data_jj_end, data_size_jj)

        touching_imin = (data_ii_start == 0)
        touching_jmin = (data_jj_start == 0)

        # Compute how much of the data slice falls outside the actual data [0,max] range
        ii_outside = size_ii - (data_ii_end - data_ii_start)  # possibly 0
        jj_outside = size_jj - (data_jj_end - data_jj_start)  # possibly 0

        if touching_imin:
            out_ii_start = ii_outside
            out_ii_end = size_ii
        else:
            out_ii_start = 0
            out_ii_end = size_ii - ii_outside
        if touching_jmin:
            out_jj_start = jj_outside
            out_jj_end = size_jj
        else:
            out_jj_start = 0
            out_jj_end = size_jj - jj_outside

    else:
        data_ii_start, out_ii_start, data_jj_start, out_jj_start = 0, 0, 0, 0
        data_ii_end, out_ii_end, data_jj_end, out_jj_end = size_ii, size_ii, size_jj, size_jj

    return [out_ii_start, out_ii_end, out_jj_start, out_jj_end, data_ii_start, data_ii_end, data_jj_start, data_jj_end]


def extract_patch_from_image(data, net, selected_input_index, settings,
                             data_ii_end, data_ii_start, data_jj_end, data_jj_start,
                             out_ii_end, out_ii_start, out_jj_end, out_jj_start, size_ii, size_jj):
    if settings.is_siamese:

        # input is first image so select first 3 channels
        if selected_input_index == 0:
            out_arr = np.zeros((3, size_ii, size_jj), dtype='float32')
            out_arr[:, out_ii_start:out_ii_end, out_jj_start:out_jj_end] = data[0:3,
                                                                           data_ii_start:data_ii_end,
                                                                           data_jj_start:data_jj_end]
        # input is second image so select second 3 channels
        elif selected_input_index == 1:
            out_arr = np.zeros((3, size_ii, size_jj), dtype='float32')
            out_arr[:, out_ii_start:out_ii_end, out_jj_start:out_jj_end] = data[3:6,
                                                                           data_ii_start:data_ii_end,
                                                                           data_jj_start:data_jj_end]
        # input is both images so select concatenate data horizontally
        elif selected_input_index == -1:
            out_arr = np.zeros((3, size_ii, size_jj * 2), dtype='float32')
            out_arr[:, out_ii_start:out_ii_end, (0 + out_jj_start):(0 + out_jj_end)] = data[0:3,
                                                                                       data_ii_start:data_ii_end,
                                                                                       data_jj_start:data_jj_end]
            out_arr[:, out_ii_start:out_ii_end, (size_jj + out_jj_start):(size_jj + out_jj_end)] = data[3:6,
                                                                                                   data_ii_start:data_ii_end,
                                                                                                   data_jj_start:data_jj_end]
        else:
            print "Error: invalid value for selected_input_index (", selected_input_index, ")"
    else:
        out_arr = np.zeros((3, size_ii, size_jj), dtype='float32')
        out_arr[:, out_ii_start:out_ii_end, out_jj_start:out_jj_end] = data[:,
                                                                       data_ii_start:data_ii_end,
                                                                       data_jj_start:data_jj_end]
    return out_arr
