from adapters.caffe_adapter import CaffeAdapter

# basic network configuration
base_folder = '%DVT_ROOT%/'
adapter = CaffeAdapter(
    deploy_prototxt_filepath=base_folder + './models/bvlc-googlenet/bvlc-googlenet-deploy.prototxt',
    network_weights_filepath=base_folder + './models/bvlc-googlenet/bvlc_googlenet.caffemodel',
    data_mean_filepath=(104, 117, 123))

# input images
static_files_dir = base_folder + './input_images/'

# UI customization
caffevis_labels          = base_folder + './models/bvlc-googlenet/ilsvrc_2012_labels.txt'
caffevis_prob_layer      = 'prob'
caffevis_label_layers    = ['loss3/classifier', 'prob']

layers_list = []
layers_list.append({'name/s': 'conv1/7x7_s2', 'format': 'normal'})
layers_list.append({'name/s': 'conv2/3x3_reduce', 'format': 'normal'})
layers_list.append({'name/s': 'conv2/3x3', 'format': 'normal'})
layers_list.append({'name/s': 'inception_3a/output', 'format': 'normal'})
layers_list.append({'name/s': 'inception_3b/output', 'format': 'normal'})
layers_list.append({'name/s': 'inception_4a/output', 'format': 'normal'})
layers_list.append({'name/s': 'inception_4b/output', 'format': 'normal'})
layers_list.append({'name/s': 'inception_4c/output', 'format': 'normal'})
layers_list.append({'name/s': 'inception_4d/output', 'format': 'normal'})
layers_list.append({'name/s': 'inception_4e/output', 'format': 'normal'})
layers_list.append({'name/s': 'inception_5a/output', 'format': 'normal'})
layers_list.append({'name/s': 'inception_5b/output', 'format': 'normal'})
layers_list.append({'name/s': 'prob', 'format': 'normal'})

def caffevis_layer_pretty_name_fn(name):
    # Shorten many layer names to fit in control pane (full layer name visible in status bar)
    name = name.replace('conv','c').replace('pool','p').replace('norm','n')
    name = name.replace('inception_','i').replace('output','o').replace('reduce','r').replace('split_','s')
    name = name.replace('__','_').replace('__','_')
    return name

# offline scripts configuration
caffevis_outputs_dir = base_folder + './models/bvlc-googlenet/outputs'
layers_to_output_in_offline_scripts = ['conv1/7x7_s2', 'conv2/3x3_reduce', 'conv2/3x3', 'inception_3a/output',
                                       'inception_3b/output', 'inception_4a/output', 'inception_4b/output',
                                       'inception_4c/output', 'inception_4d/output', 'inception_4e/output',
                                       'inception_5a/output', 'inception_5b/output', 'prob']
