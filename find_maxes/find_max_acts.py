#! /usr/bin/env python

# add parent folder to search path, to enable import of core modules like settings
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
#import ipdb as pdb
import cPickle as pickle
import numpy as np

import settings

from caffevis.caffevis_helper import set_mean
from jby_misc import WithTimer
from max_tracker import scan_images_for_maxes, scan_pairs_for_maxes


def main():
    parser = argparse.ArgumentParser(description='Finds images in a training set that cause max activation for a network; saves results in a pickled NetMaxTracker.')
    parser.add_argument('--N', type = int, default = 9, help = 'note and save top N activations')
    parser.add_argument('--gpu', action = 'store_true', default=settings.caffevis_mode_gpu, help = 'use gpu')
    parser.add_argument('--net_prototxt', type = str, default = settings.caffevis_deploy_prototxt, help = 'network prototxt to load')
    parser.add_argument('--net_weights', type = str, default = settings.caffevis_network_weights, help = 'network weights to load')
    parser.add_argument('--datadir', type = str, default = settings.static_files_dir, help = 'directory to look for files in')
    parser.add_argument('--outfile', type=str, default = settings.find_max_acts_output_file, help='output filename for pkl')
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    if args.gpu:
        caffe.set_mode_gpu()
        print 'find_max_acts mode (in main thread):     GPU'

    else:
        caffe.set_mode_cpu()
        print 'find_max_acts mode (in main thread):     CPU'

    net = caffe.Classifier(args.net_prototxt,
                           args.net_weights,
                           mean=None,
                           channel_swap=settings.caffe_net_channel_swap,
                           raw_scale=settings.caffe_net_raw_scale,
                           image_dims=settings.caffe_net_image_dims)

    data_mean = set_mean(settings.caffevis_data_mean, settings.generate_channelwise_mean, net)

    # set network batch size
    current_input_shape = net.blobs[net.inputs[0]].shape
    current_input_shape[0] = settings.max_tracker_batch_size
    net.blobs[net.inputs[0]].reshape(*current_input_shape)

    with WithTimer('Scanning images'):
        if settings.is_siamese:
            max_tracker = scan_pairs_for_maxes(settings, net, args.datadir, args.N)
        else: # normal operation
            max_tracker = scan_images_for_maxes(settings, net, args.datadir, args.N)
    with WithTimer('Saving maxes'):
        with open(args.outfile, 'wb') as ff:
            pickle.dump(max_tracker, ff, -1)



if __name__ == '__main__':
    main()
