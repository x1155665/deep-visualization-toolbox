#! /usr/bin/env python
from sys import float_info

import numpy as np
import os

from image_misc import get_tiles_height_width, caffe_load_image, ensure_uint255_and_resize_without_fit, FormattedString, \
    cv2_typeset_text, to_255
import glob



def net_preproc_forward(settings, net, img, data_hw):

    if settings.is_siamese and img.shape[2] == 6:
        appropriate_shape = data_hw + (6,)
    elif settings._calculated_is_gray_model:
        appropriate_shape = data_hw + (1,)
    else: # default is color
        appropriate_shape = data_hw + (3,)

    assert img.shape == appropriate_shape, 'img is wrong size (got %s but expected %s)' % (img.shape, appropriate_shape)
    #resized = caffe.io.resize_image(img, net.image_dims)   # e.g. (227, 227, 3)

    data_blob = net.transformer.preprocess('data', img)                # e.g. (3, 227, 227), mean subtracted and scaled to [0,255]
    data_blob = data_blob[np.newaxis,:,:,:]                   # e.g. (1, 3, 227, 227)
    output = net.forward(data=data_blob)

    return output


def get_pretty_layer_name(settings, layer_name):
    has_old_settings = hasattr(settings, 'caffevis_layer_pretty_names')
    has_new_settings = hasattr(settings, 'caffevis_layer_pretty_name_fn')
    if has_old_settings and not has_new_settings:
        print ('WARNING: Your settings.py and/or settings_model_selector.py are out of date.'
               'caffevis_layer_pretty_names has been replaced with caffevis_layer_pretty_name_fn.'
               'Update your settings.py and/or settings_model_selector.py (see documentation in'
               'setttings.py) to remove this warning.')
        return settings.caffevis_layer_pretty_names.get(layer_name, layer_name)

    ret = layer_name
    if hasattr(settings, 'caffevis_layer_pretty_name_fn'):
        ret = settings.caffevis_layer_pretty_name_fn(ret)
    if ret != layer_name:
        print '  Prettified layer name: "%s" -> "%s"' % (layer_name, ret)
    return ret


def read_label_file(filename):
    ret = []
    with open(filename, 'r') as ff:
        for line in ff:
            label = line.strip()
            if len(label) > 0:
                ret.append(label)
    return ret


def crop_to_corner(img, corner, small_padding = 1, large_padding = 2):
    '''Given an large image consisting of 3x3 squares with small_padding padding concatenated into a 2x2 grid with large_padding padding, return one of the four corners (0, 1, 2, 3)'''
    assert corner in (0,1,2,3), 'specify corner 0, 1, 2, or 3'
    assert img.shape[0] == img.shape[1], 'img is not square'
    assert img.shape[0] % 2 == 0, 'even number of pixels assumption violated'
    half_size = img.shape[0]/2
    big_ii = 0 if corner in (0,1) else 1
    big_jj = 0 if corner in (0,2) else 1
    tp = small_padding + large_padding
    #tp = 0
    return img[big_ii*half_size+tp:(big_ii+1)*half_size-tp,
               big_jj*half_size+tp:(big_jj+1)*half_size-tp]


def load_sprite_image(img_path, rows_cols, n_sprites = None):
    '''Load a 2D (3D with color channels) sprite image where
    (rows,cols) = rows_cols, slices, and returns as a 3D tensor (4D
    with color channels). Sprite shape is computed automatically. If
    n_sprites is not given, it is assumed to be rows*cols. Return as
    3D tensor with shape (n_sprites, sprite_height, sprite_width,
    sprite_channels).
    '''

    rows,cols = rows_cols
    if n_sprites is None:
        n_sprites = rows * cols
    img = caffe_load_image(img_path, color = True, as_uint = True)
    assert img.shape[0] % rows == 0, 'sprite image has shape %s which is not divisible by rows_cols %' % (img.shape, rows_cols)
    assert img.shape[1] % cols == 0, 'sprite image has shape %s which is not divisible by rows_cols %' % (img.shape, rows_cols)
    sprite_height = img.shape[0] / rows
    sprite_width  = img.shape[1] / cols
    sprite_channels = img.shape[2]

    ret = np.zeros((n_sprites, sprite_height, sprite_width, sprite_channels), dtype = img.dtype)
    for idx in xrange(n_sprites):
        # Row-major order
        ii = idx / cols
        jj = idx % cols
        ret[idx] = img[ii*sprite_height:(ii+1)*sprite_height,
                       jj*sprite_width:(jj+1)*sprite_width, :]
    return ret


def load_square_sprite_image(img_path, n_sprites):
    '''
    Just like load_sprite_image but assumes tiled image is square
    '''
    
    tile_rows,tile_cols = get_tiles_height_width(n_sprites)
    return load_sprite_image(img_path, (tile_rows, tile_cols), n_sprites = n_sprites)


def check_force_backward_true(prototxt_file):
    '''Checks whether the given file contains a line with the following text, ignoring whitespace:
    force_backward: true
    '''

    found = False
    with open(prototxt_file, 'r') as ff:
        for line in ff:
            fields = line.strip().split()
            if len(fields) == 2 and fields[0] == 'force_backward:' and fields[1] == 'true':
                found = True
                break

    if not found:
        print '\n\nWARNING: the specified prototxt'
        print '"%s"' % prototxt_file
        print 'does not contain the line "force_backward: true". This may result in backprop'
        print 'and deconv producing all zeros at the input layer. You may want to add this line'
        print 'to your prototxt file before continuing to force backprop to compute derivatives'
        print 'at the data layer as well.\n\n'


def load_mean_file(data_mean_file):
    filename, file_extension = os.path.splitext(data_mean_file)
    if file_extension == ".npy":
        # load mean from numpy array
        data_mean = np.load(data_mean_file)
        print "Loaded mean from numpy file, data_mean.shape: ", data_mean.shape

    elif file_extension == ".binaryproto":

        # load mean from binary protobuf file
        import caffe
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(data_mean_file, 'rb').read()
        blob.ParseFromString(data)
        data_mean = np.array(caffe.io.blobproto_to_array(blob))
        data_mean = np.squeeze(data_mean)
        print "Loaded mean from binaryproto file, data_mean.shape: ", data_mean.shape

    else:
        # unknown file extension, trying to load as numpy array
        data_mean = np.load(data_mean_file)
        print "Loaded mean from numpy file, data_mean.shape: ", data_mean.shape

    return data_mean

def set_mean(caffevis_data_mean, generate_channelwise_mean, net):

    if isinstance(caffevis_data_mean, basestring):
        # If the mean is given as a filename, load the file
        try:
            data_mean = load_mean_file(caffevis_data_mean)
        except IOError:
            print '\n\nCound not load mean file:', caffevis_data_mean
            print 'Ensure that the values in settings.py point to a valid model weights file, network'
            print 'definition prototxt, and mean. To fetch a default model and mean file, use:\n'
            print '$ cd models/caffenet-yos/'
            print '$ ./fetch.sh\n\n'
            raise
        input_shape = net.blobs[net.inputs[0]].data.shape[-2:]  # e.g. 227x227
        # Crop center region (e.g. 227x227) if mean is larger (e.g. 256x256)
        excess_h = data_mean.shape[1] - input_shape[0]
        excess_w = data_mean.shape[2] - input_shape[1]
        assert excess_h >= 0 and excess_w >= 0, 'mean should be at least as large as %s' % repr(input_shape)
        data_mean = data_mean[:, (excess_h / 2):(excess_h / 2 + input_shape[0]),
                          (excess_w / 2):(excess_w / 2 + input_shape[1])]
    elif caffevis_data_mean is None:
        data_mean = None
    else:
        # The mean has been given as a value or a tuple of values
        data_mean = np.array(caffevis_data_mean)
        # Promote to shape C,1,1
        # while len(data_mean.shape) < 3:
        #     data_mean = np.expand_dims(data_mean, -1)

    if generate_channelwise_mean:
        data_mean = data_mean.mean(1).mean(1)

    if data_mean is not None:
        print 'Using mean with shape:', data_mean.shape
        net.transformer.set_mean(net.inputs[0], data_mean)

    return data_mean


def get_image_from_files(settings, unit_folder_path, should_crop_to_corner, resize_shape, first_only, captions = [], values = []):
    try:

        # list unit images
        unit_images_path = sorted(glob.glob(unit_folder_path))

        mega_image = np.zeros((resize_shape[0], resize_shape[1], 3), dtype=np.uint8)

        # if no images
        if not unit_images_path:
            return mega_image

        if first_only:
            unit_images_path = [unit_images_path[0]]

        # load all images
        unit_images = [caffe_load_image(unit_image_path, color=True, as_uint=True) for unit_image_path in
                       unit_images_path]

        if settings.caffevis_clear_negative_activations:
            # clear images with 0 value
            if values:
                for i in range(len(values)):
                    if values[i] < float_info.epsilon:
                        unit_images[i] *= 0

        if should_crop_to_corner:
            unit_images = [crop_to_corner(img, 2) for img in unit_images]

        num_images = len(unit_images)
        images_per_axis = int(np.math.ceil(np.math.sqrt(num_images)))
        padding_pixel = 1

        if first_only:
            single_resized_image_shape = (resize_shape[0] - 2*padding_pixel, resize_shape[1] - 2*padding_pixel)
        else:
            single_resized_image_shape = ((resize_shape[0] / images_per_axis) - 2*padding_pixel, (resize_shape[1] / images_per_axis) - 2*padding_pixel)
        unit_images = [ensure_uint255_and_resize_without_fit(unit_image, single_resized_image_shape) for unit_image in unit_images]

        # build mega image

        should_add_caption = (len(captions) == num_images)
        defaults = {'face': settings.caffevis_score_face,
                    'fsize': settings.caffevis_score_fsize,
                    'clr': to_255(settings.caffevis_score_clr),
                    'thick': settings.caffevis_score_thick}

        for i in range(num_images):

            # add caption if we have exactly one for each image
            if should_add_caption:
                loc = settings.caffevis_score_loc[::-1]   # Reverse to OpenCV c,r order
                fs = FormattedString(captions[i], defaults)
                cv2_typeset_text(unit_images[i], [[fs]], loc)

            cell_row = i / images_per_axis
            cell_col = i % images_per_axis
            mega_image_height_start = 1 + cell_row * (single_resized_image_shape[0] + 2 * padding_pixel)
            mega_image_height_end = mega_image_height_start + single_resized_image_shape[0]
            mega_image_width_start = 1 + cell_col * (single_resized_image_shape[1] + 2 * padding_pixel)
            mega_image_width_end = mega_image_width_start + single_resized_image_shape[1]
            mega_image[mega_image_height_start:mega_image_height_end, mega_image_width_start:mega_image_width_end,:] = unit_images[i]

        return mega_image

    except:
        print '\nAttempted to load files from %s but failed. ' % unit_folder_path
        # set black image as place holder
        return np.zeros((resize_shape[0], resize_shape[1], 3), dtype=np.uint8)
        pass

    return