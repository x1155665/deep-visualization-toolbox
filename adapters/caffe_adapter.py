import sys
import os

from adapters.base_adapter import BaseAdapter
from settings_misc import replace_magic_DVT_ROOT
from caffevis.caffevis_helper import set_mean


class CaffeAdapter(BaseAdapter):

    def __init__(self, deploy_prototxt_filepath, network_weights_filepath, data_mean_filepath):

        self._deploy_prototxt_filepath = replace_magic_DVT_ROOT(deploy_prototxt_filepath)
        self._network_weights_filepath = replace_magic_DVT_ROOT(network_weights_filepath)
        self._data_mean_filepath = replace_magic_DVT_ROOT(data_mean_filepath)

        pass

    pass

    def load_network(self, settings):

        # Set the mode to CPU or GPU. Note: in the latest Caffe
        # versions, there is one Caffe object *per thread*, so the
        # mode must be set per thread! Here we set the mode for the
        # main thread; it is also separately set in CaffeProcThread.
        sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
        import caffe

        if settings.caffevis_mode_gpu:
            caffe.set_mode_gpu()
            caffe.set_device(settings.caffevis_gpu_id)
            print 'Loaded caffe in GPU mode, using device', settings.caffevis_gpu_id

        else:
            caffe.set_mode_cpu()
            print 'Loaded caffe in CPU mode'

        self._process_network_proto(settings)

        CaffeAdapter._deduce_calculated_settings_without_network(settings, self._processed_deploy_prototxt_filepath)

        net = caffe.Classifier(
            self._processed_deploy_prototxt_filepath,
            self._network_weights_filepath,
            image_dims=settings.caffe_net_image_dims,
            mean=None,  # Set to None for now, assign later
            input_scale=settings.caffe_net_input_scale,
            raw_scale=settings.caffe_net_raw_scale,
            channel_swap=settings._calculated_channel_swap)

        CaffeAdapter._deduce_calculated_settings_with_network(settings, net)

        if settings.caffe_net_transpose:
            net.transformer.set_transpose(net.inputs[0], settings.caffe_net_transpose)

        data_mean = set_mean(settings.caffevis_data_mean, settings.generate_channelwise_mean, net)

        return net, data_mean

    def _process_network_proto(self, settings):

        self._processed_deploy_prototxt_filepath = self._deploy_prototxt_filepath + ".processed_by_deepvis"

        # check if force_backwards is missing
        found_force_backwards = False
        with open(self._deploy_prototxt_filepath, 'r') as proto_file:
            for line in proto_file:
                fields = line.strip().split()
                if len(fields) == 2 and fields[0] == 'force_backward:' and fields[1] == 'true':
                    found_force_backwards = True
                    break

        # write file, adding force_backward if needed
        with open(self._deploy_prototxt_filepath, 'r') as proto_file:
            with open(self._processed_deploy_prototxt_filepath, 'w') as new_proto_file:
                if not found_force_backwards:
                    new_proto_file.write('force_backward: true\n')
                for line in proto_file:
                    new_proto_file.write(line)

        # run upgrade tool on new file name (same output file)
        upgrade_tool_command_line = settings.caffevis_caffe_root + '/build/tools/upgrade_net_proto_text.bin ' + self._processed_deploy_prototxt_filepath + ' ' + self._processed_deploy_prototxt_filepath
        os.system(upgrade_tool_command_line)

        return

    @staticmethod
    def _deduce_calculated_settings_without_network(settings, processed_deploy_prototxt_filepath):
        CaffeAdapter._set_calculated_siamese_network_format(settings)
        CaffeAdapter._set_calculated_channel_swap(settings)
        CaffeAdapter._read_network_dag(settings, processed_deploy_prototxt_filepath)

    @staticmethod
    def _deduce_calculated_settings_with_network(settings, net):
        CaffeAdapter._set_calculated_is_gray_model(settings, net)
        CaffeAdapter._set_calculated_image_dims(settings, net)

    @staticmethod
    def _set_calculated_siamese_network_format(settings):

        settings._calculated_siamese_network_format = 'normal'

        for layer_def in settings.layers_list:
            if layer_def['format'] != 'normal':
                settings._calculated_siamese_network_format = layer_def['format']
                return

    @staticmethod
    def _set_calculated_channel_swap(settings):

        if settings.caffe_net_channel_swap is not None:
            settings._calculated_channel_swap = settings.caffe_net_channel_swap

        else:
            if settings.is_siamese and settings.siamese_input_mode == 'concat_channelwise':
                settings._calculated_channel_swap = (2, 1, 0, 5, 4, 3)

            else:
                settings._calculated_channel_swap = (2, 1, 0)

    class LayerRecord:

        def __init__(self, layer_def):

            self.layer_def = layer_def
            self.name = layer_def.name
            self.type = layer_def.type

            # keep filter, stride and pad
            if layer_def.type == 'Convolution':
                self.filter = list(layer_def.convolution_param.kernel_size)
                if len(self.filter) == 1:
                    self.filter *= 2
                self.pad = list(layer_def.convolution_param.pad)
                if len(self.pad) == 0:
                    self.pad = [0, 0]
                elif len(self.pad) == 1:
                    self.pad *= 2
                self.stride = list(layer_def.convolution_param.stride)
                if len(self.stride) == 0:
                    self.stride = [1, 1]
                elif len(self.stride) == 1:
                    self.stride *= 2

            elif layer_def.type == 'Pooling':
                self.filter = [layer_def.pooling_param.kernel_size]
                if len(self.filter) == 1:
                    self.filter *= 2
                self.pad = [layer_def.pooling_param.pad]
                if len(self.pad) == 0:
                    self.pad = [0, 0]
                elif len(self.pad) == 1:
                    self.pad *= 2
                self.stride = [layer_def.pooling_param.stride]
                if len(self.stride) == 0:
                    self.stride = [1, 1]
                elif len(self.stride) == 1:
                    self.stride *= 2

            else:
                self.filter = [0, 0]
                self.pad = [0, 0]
                self.stride = [1, 1]

            # keep tops
            self.tops = list(layer_def.top)

            # keep bottoms
            self.bottoms = list(layer_def.bottom)

            # list of parent layers
            self.parents = []

            # list of child layers
            self.children = []

        pass


    @staticmethod
    def _read_network_dag(settings, processed_deploy_prototxt):
        from caffe.proto import caffe_pb2
        from google.protobuf import text_format

        # load prototxt file
        network_def = caffe_pb2.NetParameter()
        with open(processed_deploy_prototxt, 'r') as proto_file:
            text_format.Merge(str(proto_file.read()), network_def)

        # map layer name to layer record
        layer_name_to_record = dict()
        for layer_def in network_def.layer:
            if (len(layer_def.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer_def.include]):
                layer_name_to_record[layer_def.name] = CaffeAdapter.LayerRecord(layer_def)

        top_to_layers = dict()
        for layer in network_def.layer:
            # no specific phase, or TEST phase is specifically asked for
            if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
                for top in layer.top:
                    if top not in top_to_layers:
                        top_to_layers[top] = list()
                    top_to_layers[top].append(layer.name)

        # find parents and children of all layers
        for child_layer_name in layer_name_to_record.keys():
            child_layer_def = layer_name_to_record[child_layer_name]
            for bottom in child_layer_def.bottoms:
                for parent_layer_name in top_to_layers[bottom]:
                    if parent_layer_name in layer_name_to_record:
                        parent_layer_def = layer_name_to_record[parent_layer_name]
                        if parent_layer_def not in child_layer_def.parents:
                            child_layer_def.parents.append(parent_layer_def)
                        if child_layer_def not in parent_layer_def.children:
                            parent_layer_def.children.append(child_layer_def)

        # update filter, strid, pad for maxout "structures"
        for layer_name in layer_name_to_record.keys():
            layer_def = layer_name_to_record[layer_name]
            if layer_def.type == 'Eltwise' and \
               len(layer_def.parents) == 1 and \
               layer_def.parents[0].type == 'Slice' and \
               len(layer_def.parents[0].parents) == 1 and \
               layer_def.parents[0].parents[0].type in ['Convolution', 'InnerProduct']:
                layer_def.filter = layer_def.parents[0].parents[0].filter
                layer_def.stride = layer_def.parents[0].parents[0].stride
                layer_def.pad = layer_def.parents[0].parents[0].pad

        # keep helper variables in settings
        settings._network_def = network_def
        settings._layer_name_to_record = layer_name_to_record

        return

    @staticmethod
    def _set_calculated_is_gray_model(settings, net):
        if settings.is_gray_model is not None:
            settings._calculated_is_gray_model = settings.is_gray_model
        else:
            input_shape = net.blobs[net.inputs[0]].data.shape
            channels = input_shape[1]
            if channels == 1:
                settings._calculated_is_gray_model = True
            elif channels == 2 and settings.is_siamese:
                settings._calculated_is_gray_model = True
            elif channels == 3:
                settings._calculated_is_gray_model = False
            elif channels == 6 and settings.is_siamese:
                settings._calculated_is_gray_model = False
            else:
                settings._calculated_is_gray_model = None

    @staticmethod
    def _set_calculated_image_dims(settings, net):
        if settings.caffe_net_image_dims is not None:
            settings._calculated_image_dims = settings.caffe_net_image_dims
        else:
            input_shape = net.blobs[net.inputs[0]].data.shape
            settings._calculated_image_dims = input_shape[2:4]