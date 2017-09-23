
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