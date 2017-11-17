import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import cPickle as pickle

from caffe_misc import layer_name_to_top_name, get_max_data_extent
from misc import mkdir_p


def replace_magic_DVT_ROOT(path):
    dvt_root = os.path.dirname(os.path.abspath(__file__))
    return path.replace('%DVT_ROOT%', dvt_root)

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