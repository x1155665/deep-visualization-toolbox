import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import cPickle as pickle

from caffevis.caffevis_helper import set_mean
from caffe_misc import layer_name_to_top_name, get_max_data_extent
from misc import mkdir_p

def deduce_calculated_settings_without_network(settings):
    set_calculated_siamese_network_format(settings)
    set_calculated_channel_swap(settings)
    read_network_dag(settings)


def deduce_calculated_settings_with_network(settings, net):
    set_calculated_is_gray_model(settings, net)
    set_calculated_image_dims(settings, net)


def set_calculated_is_gray_model(settings, net):
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


def set_calculated_image_dims(settings, net):
    if settings.caffe_net_image_dims is not None:
        settings._calculated_image_dims = settings.caffe_net_image_dims
    else:
        input_shape = net.blobs[net.inputs[0]].data.shape
        settings._calculated_image_dims = input_shape[2:4]


def set_calculated_siamese_network_format(settings):

    settings._calculated_siamese_network_format = 'normal'

    for layer_def in settings.layers_list:
        if layer_def['format'] != 'normal':
            settings._calculated_siamese_network_format = layer_def['format']
            return


def set_calculated_channel_swap(settings):

    if settings.caffe_net_channel_swap is not None:
        settings._calculated_channel_swap = settings.caffe_net_channel_swap

    else:
        if settings.is_siamese and settings.siamese_input_mode == 'concat_channelwise':
            settings._calculated_channel_swap = (2, 1, 0, 5, 4, 3)

        else:
            settings._calculated_channel_swap = (2, 1, 0)


def process_network_proto(settings):

    settings._processed_deploy_prototxt = settings.caffevis_deploy_prototxt + ".processed_by_deepvis"

    # check if force_backwards is missing
    found_force_backwards = False
    with open(settings.caffevis_deploy_prototxt, 'r') as proto_file:
        for line in proto_file:
            fields = line.strip().split()
            if len(fields) == 2 and fields[0] == 'force_backward:' and fields[1] == 'true':
                found_force_backwards = True
                break

    # write file, adding force_backward if needed
    with open(settings.caffevis_deploy_prototxt, 'r') as proto_file:
        with open(settings._processed_deploy_prototxt, 'w') as new_proto_file:
            if not found_force_backwards:
                new_proto_file.write('force_backward: true\n')
            for line in proto_file:
                new_proto_file.write(line)

    # run upgrade tool on new file name (same output file)
    upgrade_tool_command_line = settings.caffevis_caffe_root + '/build/tools/upgrade_net_proto_text.bin ' + settings._processed_deploy_prototxt + ' ' + settings._processed_deploy_prototxt
    os.system(upgrade_tool_command_line)

    return


def load_network(settings):

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

    process_network_proto(settings)

    deduce_calculated_settings_without_network(settings)

    net = caffe.Classifier(
        settings._processed_deploy_prototxt,
        settings.caffevis_network_weights,
        image_dims=settings.caffe_net_image_dims,
        mean=None,  # Set to None for now, assign later
        input_scale=settings.caffe_net_input_scale,
        raw_scale=settings.caffe_net_raw_scale,
        channel_swap=settings._calculated_channel_swap)

    deduce_calculated_settings_with_network(settings, net)

    if settings.caffe_net_transpose:
        net.transformer.set_transpose(net.inputs[0], settings.caffe_net_transpose)

    data_mean = set_mean(settings.caffevis_data_mean, settings.generate_channelwise_mean, net)

    return net, data_mean


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


def read_network_dag(settings):
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format

    # load prototxt file
    network_def = caffe_pb2.NetParameter()
    with open(settings._processed_deploy_prototxt, 'r') as proto_file:
        text_format.Merge(str(proto_file.read()), network_def)

    # map layer name to layer record
    layer_name_to_record = dict()
    for layer_def in network_def.layer:
        if (len(layer_def.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer_def.include]):
            layer_name_to_record[layer_def.name] = LayerRecord(layer_def)

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


def _get_receptive_fields_cache_filename(settings):
    return os.path.join(settings.caffevis_outputs_dir, 'receptive_fields_cache.pickled')

def get_receptive_field(settings, net, layer_name):

    # flag which indicates whether the dictionary was changed hence we need to write it to cache
    should_save_to_cache = False

    # check if dictionary exists
    if not hasattr(settings, '_receptive_field_per_layer'):

        # if it doesn't, try load it from file
        receptive_fields_cache_filename = _get_receptive_fields_cache_filename(settings)
        if os.path.isfile(receptive_fields_cache_filename):
            try:
                with open(receptive_fields_cache_filename, 'rb') as receptive_fields_cache_file:
                    settings._receptive_field_per_layer = pickle.load(receptive_fields_cache_file)
            except:
                settings._receptive_field_per_layer = dict()
                should_save_to_cache = True
        else:
            settings._receptive_field_per_layer = dict()
            should_save_to_cache = True

    # calculate lazy
    if not settings._receptive_field_per_layer.has_key(layer_name):
        print "Calculating receptive fields for layer %s" % (layer_name)
        top_name = layer_name_to_top_name(net, layer_name)
        if top_name is not None:
            blob = net.blobs[top_name].data
            is_spatial = (len(blob.shape) == 4)
            layer_receptive_field = get_max_data_extent(net, settings, layer_name, is_spatial)
            settings._receptive_field_per_layer[layer_name] = layer_receptive_field
            should_save_to_cache = True

    if should_save_to_cache:
        try:
            receptive_fields_cache_filename = _get_receptive_fields_cache_filename(settings)
            mkdir_p(settings.caffevis_outputs_dir)
            with open(receptive_fields_cache_filename, 'wb') as receptive_fields_cache_file:
                pickle.dump(settings._receptive_field_per_layer, receptive_fields_cache_file, -1)
        except IOError:
            # ignore problems in cache saving
            pass

    return settings._receptive_field_per_layer[layer_name]