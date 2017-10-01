
# basic network configuration
base_folder = '%DVT_ROOT%/'
caffevis_deploy_prototxt = base_folder + './models/squeezenet/deploy.prototxt'
caffevis_network_weights = base_folder + './models/squeezenet/squeezenet_v1.0.caffemodel'
caffevis_data_mean       = (104, 117, 123)

# input images
static_files_dir = base_folder + './input_images/'

# UI customization
caffevis_labels          = base_folder + './models/squeezenet/ilsvrc_2012_labels.txt'
caffevis_prob_layer      = 'prob'
caffevis_label_layers    = ['conv10', 'relu_conv10', 'pool10', 'prob']

def caffevis_layer_pretty_name_fn(name):
    name = name.replace('fire','f').replace('relu_expand','re').replace('expand','e').replace('concat','c').replace('squeeze','s')
    name = name.replace('1x1_','').replace('1x1','')
    return name

# Don't display duplicate *_split_* layers
caffevis_filter_layers = lambda name: '_split_' in name

# offline scripts configuration
caffevis_outputs_dir = base_folder + './models/squeezenet/outputs'
