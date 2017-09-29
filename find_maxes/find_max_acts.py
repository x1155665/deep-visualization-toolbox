#! /usr/bin/env python

# this import must comes first to make sure we use the non-display backend
import matplotlib
matplotlib.use('Agg')

# add parent folder to search path, to enable import of core modules like settings
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
import cPickle as pickle
import numpy as np

import settings

from caffevis.caffevis_helper import set_mean
from jby_misc import WithTimer
from max_tracker import scan_images_for_maxes, scan_pairs_for_maxes
from settings_misc import load_network

from misc import mkdir_p

def pickle_to_text(pickle_filename):

    with open(pickle_filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    data_dict = data.__dict__.copy()

    with open(pickle_filename + ".txt", 'wt') as text_file:
        text_file.write(str(data_dict))

    return

def main():

    parser = argparse.ArgumentParser(description='Finds images in a training set that cause max activation for a network; saves results in a pickled NetMaxTracker.')
    parser.add_argument('--N', type = int, default = 9, help = 'note and save top N activations')
    parser.add_argument('--gpu', action = 'store_true', default = settings.caffevis_mode_gpu, help = 'use gpu')
    parser.add_argument('--net_prototxt', type = str, default = settings.caffevis_deploy_prototxt, help = 'network prototxt to load')
    parser.add_argument('--net_weights', type = str, default = settings.caffevis_network_weights, help = 'network weights to load')
    parser.add_argument('--datadir', type = str, default = settings.static_files_dir, help = 'directory to look for files in')
    parser.add_argument('--outfile', type=str, default = os.path.join(settings.caffevis_outputs_dir, 'find_max_acts_output.pickled'), help='output filename for pkl')
    parser.add_argument('--outdir', type = str, default = settings.caffevis_outputs_dir, help = 'Which output directory to use. Files are output into outdir/layer/unit_%%04d/{max_histogram}.png')
    parser.add_argument('--do-histograms', action = 'store_true', default = settings.max_tracker_do_histograms, help = 'Output histogram image file containing histogrma of max values per channel')
    parser.add_argument('--do-correlation', action = 'store_true', default = settings.max_tracker_do_correlation, help = 'Output correlation image file containing correlation of channels per layer')

    args = parser.parse_args()

    settings.caffevis_deploy_prototxt = args.net_prototxt
    settings.caffevis_network_weights = args.net_weights

    net, data_mean = load_network(settings)

    # validate batch size
    if settings.is_siamese and settings._calculated_siamese_network_format == 'siamese_batch_pair':
        # currently, no batch support for siamese_batch_pair networks
        # it can be added by simply handle the batch indexes properly, but it should be thoroughly tested
        assert (settings.max_tracker_batch_size == 1)

    # set network batch size
    current_input_shape = net.blobs[net.inputs[0]].shape
    current_input_shape[0] = settings.max_tracker_batch_size
    net.blobs[net.inputs[0]].reshape(*current_input_shape)
    net.reshape()

    with WithTimer('Scanning images'):
        if settings.is_siamese:
            net_max_tracker = scan_pairs_for_maxes(settings, net, args.datadir, args.N, args.outdir)
        else: # normal operation
            net_max_tracker = scan_images_for_maxes(settings, net, args.datadir, args.N, args.outdir)

    save_max_tracker_to_file(args.outfile, net_max_tracker)

    if args.do_correlation:
        net_max_tracker.calculate_correlation(args.outdir)

    if args.do_histograms:
        net_max_tracker.calculate_histograms(args.outdir)


def save_max_tracker_to_file(filename, net_max_tracker):

    dir_name = os.path.dirname(filename)
    mkdir_p(dir_name)

    with WithTimer('Saving maxes'):
        with open(filename, 'wb') as ff:
            pickle.dump(net_max_tracker, ff, -1)
        # save text version of pickle file for easier debugging
        pickle_to_text(filename)


def load_max_tracker_from_file(filename):

    import max_tracker
    # load pickle file
    with open(filename, 'rb') as tracker_file:
        net_max_tracker = pickle.load(tracker_file)

    return net_max_tracker


if __name__ == '__main__':
    main()
