
# basic network configuration
base_folder = '%DVT_ROOT%/'
caffevis_deploy_prototxt = base_folder + './models/caffenet-yos/caffenet-yos-deploy.prototxt'
caffevis_network_weights = base_folder + './models/caffenet-yos/caffenet-yos-weights'
caffevis_data_mean       = base_folder + './models/caffenet-yos/ilsvrc_2012_mean.npy'

# input images
static_files_dir = base_folder + './input_images/'

# UI customization
caffevis_label_layers    = ['fc8', 'prob']
caffevis_labels          = base_folder + './models/caffenet-yos/ilsvrc_2012_labels.txt'
caffevis_prob_layer      = 'prob'

def caffevis_layer_pretty_name_fn(name):
    return name.replace('pool','p').replace('norm','n')

# offline scripts configuration
# caffevis_outputs_dir = base_folder + './models/caffenet-yos/unit_jpg_vis'
caffevis_outputs_dir = base_folder + './models/caffenet-yos/outputs'
layers_to_output_in_offline_scripts = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']
