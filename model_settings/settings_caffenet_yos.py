from adapters.caffe_adapter import CaffeAdapter

base_folder = '%DVT_ROOT%/'
adapter = CaffeAdapter(
    deploy_prototxt_filepath=base_folder + './models/caffenet-yos/caffenet-yos-deploy.prototxt',
    network_weights_filepath=base_folder + './models/caffenet-yos/caffenet-yos-weights',
    data_mean_ref=base_folder + './models/caffenet-yos/ilsvrc_2012_mean.npy')

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
