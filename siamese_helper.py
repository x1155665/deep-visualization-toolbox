import os
# import numpy as np
from numpy import expand_dims, concatenate
from caffe_misc import layer_name_to_top_name
from image_misc import resize_without_fit

class SiameseInputMode:
    FIRST_IMAGE = 0
    SECOND_IMAGE = 1
    BOTH_IMAGES = 2
    NUMBER_OF_MODES = 3

class SiameseHelper(object):
    '''helper class for handling all sorts of operations related to siamese networks
    this class should encapsulate the different types of siamese network implementation'''

    def __init__(self, layers_list):

        # define class members
        self.layers_list = layers_list
        self.layer_name_to_normalized_layer_name = dict()
        self.normalized_layer_name_to_denormalized_layer_name = dict()
        self.layer_name_to_index_of_saved_image = dict()

        # init dictionaries
        self._init_layer_name_to_normalized_layer_name()
        self._init_normalized_layer_name_to_denormalized_layer_name()
        self._init_layer_name_to_index_of_saved_image()

        return

    def _init_layer_name_to_normalized_layer_name(self):
        '''
        init layer_name_to_normalized_layer_name dictionary
        :return: none
        '''

        for layer_def in self.layers_list:

            layer_format = layer_def['format']
            layer_names = layer_def['name/s']

            if layer_format == 'normal':
                self.layer_name_to_normalized_layer_name[layer_names] = layer_names

            elif layer_format == 'siamese_layer_pair':
                self.layer_name_to_normalized_layer_name[layer_names[0]] = layer_names[0]
                self.layer_name_to_normalized_layer_name[layer_names[1]] = layer_names[0]

            elif layer_format == 'siamese_batch_pair':
                self.layer_name_to_normalized_layer_name[layer_names] = layer_names

        return

    def _init_normalized_layer_name_to_denormalized_layer_name(self):
        '''
        init normalized_layer_name_to_denormalized_layer_name dictionary
        :return: none
        '''

        for layer_def in self.layers_list:

            layer_format = layer_def['format']
            layer_names = layer_def['name/s']

            if layer_format == 'normal':
                self.normalized_layer_name_to_denormalized_layer_name[layer_names] = layer_names

            elif layer_format == 'siamese_layer_pair':
                self.normalized_layer_name_to_denormalized_layer_name[layer_names[1]] = layer_names[1]
                self.normalized_layer_name_to_denormalized_layer_name[layer_names[0]] = layer_names[1]

            elif layer_format == 'siamese_batch_pair':
                self.normalized_layer_name_to_denormalized_layer_name[layer_names] = layer_names

        return

    def _init_layer_name_to_index_of_saved_image(self):
        '''
        init layer_name_to_index_of_saved_image dictionary
        :return: none
        '''

        for layer_def in self.layers_list:

            layer_format = layer_def['format']
            layer_names = layer_def['name/s']

            if layer_format == 'normal':
                self.layer_name_to_index_of_saved_image[layer_names] = -1

            elif layer_format == 'siamese_layer_pair':
                self.layer_name_to_index_of_saved_image[layer_names[0]] = 0
                self.layer_name_to_index_of_saved_image[layer_names[1]] = 1

            elif layer_format == 'siamese_batch_pair':
                # raise NotImplementedError()
                self.layer_name_to_index_of_saved_image[layer_names] = -1

        return

    def normalize_layer_name_for_max_tracker(self, layer_name):
        '''
        function used to normalize layer name, e.g. 'conv1' and 'conv1_p' will be normalized to 'conv1', given suitable
        layers_list setting.
        :param layer_name: layer name to normalize
        :return: normalized layer name
        '''

        if self.layer_name_to_normalized_layer_name.has_key(layer_name):
            return self.layer_name_to_normalized_layer_name[layer_name]

        return layer_name

    def denormalize_layer_name_for_max_tracker(self, normalized_layer_name, selected_input_index):
        '''
        function which returns the denormalized form of the layer name, given the normalized layer name and selected
        input index which should be 0 or 1
        e.g. denormalize_layer_name_for_max_tracker('conv1', 1) == 'conv1_p', given suitable layers_list setting
        :param normalized_layer_name: normalized layer name
        :param selected_input_index: selected input index, 0 or 1
        :return: the denormalized layer name
        '''

        if selected_input_index == 0:
            return normalized_layer_name

        if self.normalized_layer_name_to_denormalized_layer_name.has_key(normalized_layer_name):
            return self.normalized_layer_name_to_denormalized_layer_name[normalized_layer_name]

        # this can happen for layer names which don't appear in the layers_list setting
        return normalized_layer_name

    def get_index_of_saved_image_by_layer_name(self, layer_name):
        '''
        function which returns the index of image to save (0 or 1) given the layer name
        e.g. for conv1_p returns 1, for 'conv1' returns 0
        the decision is done using the layers_list setting used in max tracker
        :param layer_name: layer name
        :return index of image in the pair, 0 or 1:
        '''

        if self.layer_name_to_index_of_saved_image.has_key(layer_name):
            return self.layer_name_to_index_of_saved_image[layer_name]

        # this can happen for layer names which don't appear in the layers_list setting
        return -1

    @staticmethod
    def get_header_from_layer_def(layer_def):
        '''
        helper function which returns the header name, given a single layer, or layer pair
        :param layer: can be either single layer string, or apir of layers
        :return: header for layer
        '''

        # if we have only a single layer, the header is the layer name

        if layer_def['format'] == 'normal':
            return layer_def['name/s']

        elif layer_def['format'] == 'siamese_layer_pair':
            # build header in format: common_prefix + first_postfix | second_postfix
            prefix = os.path.commonprefix(layer_def['name/s'])
            prefix_len = len(prefix)
            postfix0 = layer_def['name/s'][0][prefix_len:]
            postfix1 = layer_def['name/s'][1][prefix_len:]
            header_name = '%s%s|%s' % (prefix, postfix0, postfix1)
            return header_name

        elif layer_def['format'] == 'siamese_batch_pair':
            return layer_def['name/s']

    @staticmethod
    def get_default_layer_name(layer_def):
        '''
        get layer name when the caller needs some 'default' choice
        :param layer_def: layer definition object
        :return: default_layer_name
        '''

        if layer_def['format'] == 'normal':
            default_layer_name = layer_def['name/s']

        elif layer_def['format'] == 'siamese_layer_pair':
            default_layer_name = layer_def['name/s'][0]

        elif layer_def['format'] == 'siamese_batch_pair':
            default_layer_name = layer_def['name/s']

        else:
            raise Exception("get_default_layer_name() got invalid layer_def['format']=%s" % layer_def['format'])

        return default_layer_name

    @staticmethod
    def get_single_selected_layer_name(layer_def, siamese_input_mode):

        if layer_def['format'] == 'normal':
            return layer_def['name/s']

        elif layer_def['format'] == 'siamese_layer_pair':
            if siamese_input_mode == SiameseInputMode.FIRST_IMAGE:
                return layer_def['name/s'][0]
            elif siamese_input_mode == SiameseInputMode.SECOND_IMAGE:
                return layer_def['name/s'][1]
            else:
                raise Exception('in get_single_selected_blob() siamese_input_mode cant be BOTH')

        elif layer_def['format'] == 'siamese_batch_pair':
            return layer_def['name/s']

        else:
            raise Exception("get_single_selected_blob() got invalid layer_def['format']=%s" % layer_def['format'])

    @staticmethod
    def _get_single_selected_blob(net, layer_def, siamese_input_mode, blob_selector):
        '''
        function used to extract the single selected blob according to the specified layer and siamese input mode and
        blob selector.
        note that it is invalid to call this function when siamese input mode is BOTH
        this is the main function which contains logic on the siamese network internal format structure
        :param net: network containing the blob to extract
        :param layer_def: layer requested
        :param siamese_input_mode: siamese input mode
        :param blob_selector: lambda function which lets us choose between data and diff blobs
        :return: requested single blob
        '''

        if layer_def['format'] == 'normal':
            return blob_selector(net.blobs[layer_name_to_top_name(net, layer_def['name/s'])])[0]

        elif layer_def['format'] == 'siamese_layer_pair':
            if siamese_input_mode == SiameseInputMode.FIRST_IMAGE:
                selected_layer_name = layer_def['name/s'][0]
            elif siamese_input_mode == SiameseInputMode.SECOND_IMAGE:
                selected_layer_name = layer_def['name/s'][1]
            else:
                raise Exception('in get_single_selected_blob() siamese_input_mode cant be BOTH')
            return blob_selector(net.blobs[layer_name_to_top_name(net, selected_layer_name)])[0]

        elif layer_def['format'] == 'siamese_batch_pair':
            if siamese_input_mode == SiameseInputMode.FIRST_IMAGE:
                selected_batch_index = 0
            elif siamese_input_mode == SiameseInputMode.SECOND_IMAGE:
                selected_batch_index = 1
            else:
                raise Exception('in get_single_selected_blob() siamese_input_mode cant be BOTH')
            return blob_selector(net.blobs[layer_name_to_top_name(net, layer_def['name/s'])])[selected_batch_index]

        else:
            raise Exception("get_single_selected_blob() got invalid layer_def['format']=%s" % layer_def['format'])


    @staticmethod
    def get_single_selected_data_blob(net, layer_def, siamese_input_mode):
        '''
        function used to extract the single selected DATA blob according to the specified layer and siamese input mode
         note that it is invalid to call this function when siamese input mode is BOTH
        :param net: network containing the blob to extract
        :param layer_def: layer requested
        :param siamese_input_mode: siamese input mode
        :return: requested single data blob
        '''

        return SiameseHelper._get_single_selected_blob(net, layer_def, siamese_input_mode, blob_selector=lambda layer_object: layer_object.data)

    @staticmethod
    def get_single_selected_diff_blob(net, layer_def, siamese_input_mode):
        '''
        function used to extract the single selected DIFF blob according to the specified layer and siamese input mode
         note that it is invalid to call this function when siamese input mode is BOTH
        :param net: network containing the blob to extract
        :param layer_def: layer requested
        :param siamese_input_mode: siamese input mode
        :return: requested single diff blob
        '''

        return SiameseHelper._get_single_selected_blob(net, layer_def, siamese_input_mode, blob_selector=lambda layer_object: layer_object.diff)

    @staticmethod
    def _get_siamese_selected_blobs(net, layer_def, siamese_input_mode, blob_selector):
        '''
        function used to extract both blobs according to the specified layer and siamese input mode and
        blob selector.
        this is the main function which contains logic on the siamese network internal format structure
        :param net: network containing the blob to extract
        :param layer_def: layer requested
        :param siamese_input_mode: siamese input mode
        :param blob_selector:
        :return: first_blob, second_blob
        '''

        if layer_def['format'] == 'normal':
            raise Exception('function get_siamese_blobs() should not be called when layer is in normal format')

        elif layer_def['format'] == 'siamese_layer_pair':
            return blob_selector(net.blobs[layer_name_to_top_name(net, layer_def['name/s'][0])])[0], blob_selector(net.blobs[layer_name_to_top_name(net, layer_def['name/s'][1])])[0]

        elif layer_def['format'] == 'siamese_batch_pair':
            return blob_selector(net.blobs[layer_name_to_top_name(net, layer_def['name/s'])])[0], blob_selector(net.blobs[layer_name_to_top_name(net, layer_def['name/s'])])[1]

        else:
            raise Exception("get_siamese_blobs() got invalid layer_def['format']=%s" % layer_def['format'])

    @staticmethod
    def get_siamese_selected_data_blobs(net, layer_def, siamese_input_mode):
        '''
        function used to extract both DATA blobs according to the specified layer and siamese input mode
         note that it is invalid to call this function when siamese input mode is not BOTH
        :param net: network containing the blob to extract
        :param layer_def: layer requested
        :param siamese_input_mode: siamese input mode
        :return: first_blob, second_blob
        '''

        return SiameseHelper._get_siamese_selected_blobs(net, layer_def, siamese_input_mode, blob_selector=lambda layer_object: layer_object.data)

    @staticmethod
    def get_siamese_selected_diff_blobs(net, layer_def, siamese_input_mode):
        '''
        function used to extract both DIFF blobs according to the specified layer and siamese input mode
         note that it is invalid to call this function when siamese input mode is not BOTH
        :param net: network containing the blob to extract
        :param layer_def: layer requested
        :param siamese_input_mode: siamese input mode
        :return: first_blob, second_blob
        '''

        return SiameseHelper._get_siamese_selected_blobs(net, layer_def, siamese_input_mode, blob_selector=lambda layer_object: layer_object.diff)

    @staticmethod
    def is_pair_of_layers(layer_def):

        return layer_def['format'] in ['siamese_layer_pair', 'siamese_batch_pair']

    @staticmethod
    def siamese_input_mode_has_two_images(layer_def, siamese_input_mode):
        '''
        helper function which checks whether the input mode is two images, and the provided layer contains two layer names
        :param layer: can be a single string layer name, or a pair of layer names
        :return: True if both the input mode is BOTH_IMAGES and layer contains two layer names, False oherwise
        '''

        return siamese_input_mode == SiameseInputMode.BOTH_IMAGES and SiameseHelper.is_pair_of_layers(layer_def)

    @staticmethod
    def backward_from_layer(net, backprop_layer_def, backprop_unit, siamese_input_mode):

        # if we are in siamese_batch_pair, we don't care of siamese_input_mode since we must do deconv on the 2-batch
        # otherwise, if we are in siamese_layer_pair, we do it on both layers only if backprop deconv are requested
        if (backprop_layer_def['format'] == 'siamese_batch_pair') or \
            (backprop_layer_def['format'] == 'siamese_layer_pair' and siamese_input_mode == SiameseInputMode.BOTH_IMAGES):

            diffs0, diffs1 = SiameseHelper.get_siamese_selected_diff_blobs(net, backprop_layer_def, siamese_input_mode)
            diffs0, diffs1 = diffs0 * 0, diffs1 * 0
            data0, data1 = SiameseHelper.get_siamese_selected_data_blobs(net, backprop_layer_def, siamese_input_mode)
            diffs0[backprop_unit], diffs1[backprop_unit] = data0[backprop_unit], data1[backprop_unit]

            # add batch dimension
            diffs0 = expand_dims(diffs0, 0)
            diffs1 = expand_dims(diffs1, 0)

            if backprop_layer_def['format'] == 'siamese_layer_pair':
                net.backward_from_layer(backprop_layer_def['name/s'][0], diffs0, zero_higher=True)
                net.backward_from_layer(backprop_layer_def['name/s'][1], diffs1, zero_higher=True)

            elif backprop_layer_def['format'] == 'siamese_batch_pair':
                # combine them to 2-batch and send once
                diffs = concatenate((diffs0, diffs1), axis=0)
                net.backward_from_layer(backprop_layer_def['name/s'], diffs, zero_higher=True)
        else:

            diffs = SiameseHelper.get_single_selected_diff_blob(net, backprop_layer_def, siamese_input_mode)
            diffs = diffs * 0
            data = SiameseHelper.get_single_selected_data_blob(net, backprop_layer_def, siamese_input_mode)
            diffs[backprop_unit] = data[backprop_unit]

            # add batch dimension
            diffs = expand_dims(diffs, 0)

            selected_backprop_layer_name = SiameseHelper.get_single_selected_layer_name(backprop_layer_def, siamese_input_mode)
            net.backward_from_layer(selected_backprop_layer_name, diffs, zero_higher=True)

        pass

    @staticmethod
    def deconv_from_layer(net, backprop_layer_def, backprop_unit, siamese_input_mode, deconv_type):

        # if we are in siamese_batch_pair, we don't care of siamese_input_mode since we must do deconv on the 2-batch
        # otherwise, if we are in siamese_layer_pair, we do it on both layers only if both deconv are requested
        if (backprop_layer_def['format'] == 'siamese_batch_pair') or \
            (backprop_layer_def['format'] == 'siamese_layer_pair' and siamese_input_mode == SiameseInputMode.BOTH_IMAGES):

            diffs0, diffs1 = SiameseHelper.get_siamese_selected_diff_blobs(net, backprop_layer_def, siamese_input_mode)
            diffs0, diffs1 = diffs0 * 0, diffs1 * 0
            data0, data1 = SiameseHelper.get_siamese_selected_data_blobs(net, backprop_layer_def, siamese_input_mode)
            diffs0[backprop_unit], diffs1[backprop_unit] = data0[backprop_unit], data1[backprop_unit]

            # add batch dimension
            diffs0 = expand_dims(diffs0, 0)
            diffs1 = expand_dims(diffs1, 0)

            if backprop_layer_def['format'] == 'siamese_layer_pair':
                net.deconv_from_layer(backprop_layer_def['name/s'][0], diffs0, zero_higher=True, deconv_type=deconv_type)
                net.deconv_from_layer(backprop_layer_def['name/s'][1], diffs1, zero_higher=True, deconv_type=deconv_type)

            elif backprop_layer_def['format'] == 'siamese_batch_pair':
                # combine them to 2-batch and send once
                diffs = concatenate((diffs0, diffs1), axis=0)
                net.deconv_from_layer(backprop_layer_def['name/s'], diffs, zero_higher=True, deconv_type=deconv_type)

        else:

            diffs = SiameseHelper.get_single_selected_diff_blob(net, backprop_layer_def, siamese_input_mode)
            diffs = diffs * 0
            data = SiameseHelper.get_single_selected_data_blob(net, backprop_layer_def, siamese_input_mode)
            diffs[backprop_unit] = data[backprop_unit]

            # add batch dimension
            diffs = expand_dims(diffs, 0)

            selected_backprop_layer_name = SiameseHelper.get_single_selected_layer_name(backprop_layer_def, siamese_input_mode)
            net.deconv_from_layer(selected_backprop_layer_name, diffs, zero_higher=True, deconv_type=deconv_type)

    @staticmethod
    def get_image_from_frame(frame, is_siamese, image_shape, siamese_input_mode):

        if is_siamese and ((type(frame),len(frame)) == (tuple,2)):

            if siamese_input_mode == SiameseInputMode.BOTH_IMAGES:
                frame1 = frame[0]
                frame2 = frame[1]
                half_pane_shape = (image_shape[0], image_shape[1]/2)
                frame_disp1 = resize_without_fit(frame1[:], half_pane_shape)
                frame_disp2 = resize_without_fit(frame2[:], half_pane_shape)
                frame_disp = concatenate((frame_disp1, frame_disp2), axis=1)
            elif siamese_input_mode == SiameseInputMode.FIRST_IMAGE:
                frame_disp = resize_without_fit(frame[0], image_shape)
            elif siamese_input_mode == SiameseInputMode.SECOND_IMAGE:
                frame_disp = resize_without_fit(frame[1], image_shape)

        else:
            frame_disp = resize_without_fit(frame[:], image_shape)

        return frame_disp