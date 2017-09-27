import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from caffevis.caffevis_helper import set_mean


def deduce_calculated_settings(settings, net):
    set_calculated_is_gray_model(settings, net)
    set_calculated_image_dims(settings, net)
    read_network_dag(settings)


def set_calculated_is_gray_model(settings, net):
    if settings.is_gray_model is not None:
        settings._calculated_is_gray_model = settings.is_gray_model
    else:
        input_shape = net.blobs[net.inputs[0]].data.shape
        channels = input_shape[1]
        if channels == 1:
            settings._calculated_is_gray_model = True
        elif channels == 2 and setings.is_siamese:
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


def get_layer_info(settings, layer_name):
    '''
        get layer info (name, type, input, output, filter, stride, pad) from settings

    :param settings: contains script settings
    :param layer_name: name of layer
    :return: current_layer and previous_layer tuples of (name, type, input, output, filter, stride, pad)
    '''

    # go over layers

    previous_layer = (None, None, None, None, None, None, None)
    current_layer = (None, None, None, None, None, None, None)
    for (name, type, input, output, filter, stride, pad) in settings.max_tracker_layers_list:
        if name == layer_name:
            current_layer = (name, type, input, output, filter, stride, pad)
            return current_layer, previous_layer

        # update previous layer
        previous_layer = (name, type, input, output, filter, stride, pad)

    return current_layer, previous_layer


def load_network(settings):

    # Set the mode to CPU or GPU. Note: in the latest Caffe
    # versions, there is one Caffe object *per thread*, so the
    # mode must be set per thread! Here we set the mode for the
    # main thread; it is also separately set in CaffeProcThread.
    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe
    if settings.caffevis_mode_gpu:
        caffe.set_mode_gpu()
        print 'CaffeVisApp mode (in main thread):     GPU'
    else:
        caffe.set_mode_cpu()
        print 'CaffeVisApp mode (in main thread):     CPU'

    net = caffe.Classifier(
        settings.caffevis_deploy_prototxt,
        settings.caffevis_network_weights,
        image_dims=settings.caffe_net_image_dims,
        mean=None,  # Set to None for now, assign later
        input_scale=settings.caffe_net_input_scale,
        raw_scale=settings.caffe_net_raw_scale,
        channel_swap=settings.caffe_net_channel_swap)

    deduce_calculated_settings(settings, net)

    if settings.caffe_net_transpose:
        net.transformer.set_transpose(net.inputs[0], settings.caffe_net_transpose)

    data_mean = set_mean(settings.caffevis_data_mean, settings.generate_channelwise_mean, net)

    return net, data_mean


def read_network_dag(settings):
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format

    # load prototxt file
    network_def = caffe_pb2.NetParameter()
    f = open(settings.caffevis_deploy_prototxt, 'r')
    text_format.Merge(str(f.read()), network_def)

    # map blob name to list of layers having this blob name as input
    bottom_to_layers = dict()
    for layer in network_def.layer:
        # no specific phase, or TEST phase is specifically asked for
        if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
            for bottom in layer.bottom:
                if bottom not in bottom_to_layers:
                    bottom_to_layers[bottom] = list()
                bottom_to_layers[bottom].append(layer.name)

    top_to_layers = dict()
    for layer in network_def.layer:
        # no specific phase, or TEST phase is specifically asked for
        if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
            for top in layer.top:
                if top not in top_to_layers:
                    top_to_layers[top] = list()
                top_to_layers[top].append(layer.name)
    if 'data' not in top_to_layers:
        top_to_layers['data'] = ['data']

    layer_to_tops = dict()
    for layer in network_def.layer:
        if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
            for top in layer.top:
                if layer.name not in layer_to_tops:
                    layer_to_tops[layer.name] = list()
                layer_to_tops[layer.name].append(top)
    if 'data' not in layer_to_tops:
        layer_to_tops['data'] = ['data']

    layer_to_bottoms = dict()
    for layer in network_def.layer:
        if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
            for bottom in layer.bottom:
                if layer.name not in layer_to_bottoms:
                    layer_to_bottoms[layer.name] = list()
                layer_to_bottoms[layer.name].append(bottom)
    # if 'data' not in layer_to_bottoms:
    #     layer_to_bottoms['data'] = ['data']

    inplace_layers = list()
    for layer in network_def.layer:
        if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
            if len(layer.top) == 1 and len(layer.bottom) == 1 and layer.top[0] == layer.bottom[0]:
                inplace_layers.append(layer.name)

    layer_name_to_def = dict()
    for layer in network_def.layer:
        if (len(layer.include) == 0) or (caffe_pb2.TEST in [item.phase for item in layer.include]):
            layer_name_to_def[layer.name] = layer

    # keep helper variables in settings
    settings._network_def = network_def
    settings._bottom_to_layers = bottom_to_layers
    settings._top_to_layers = top_to_layers
    settings._layer_to_tops = layer_to_tops
    settings._layer_to_bottoms = layer_to_bottoms
    settings._inplace_layers = inplace_layers
    settings._layer_name_to_def = layer_name_to_def

    return
