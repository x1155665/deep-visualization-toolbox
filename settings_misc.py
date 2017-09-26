import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from caffevis.caffevis_helper import set_mean

def deduce_calculated_settings(settings, net):
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
