
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
