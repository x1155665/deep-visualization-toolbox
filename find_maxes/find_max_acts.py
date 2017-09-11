#! /usr/bin/env python

import argparse
import ipdb as pdb
import cPickle as pickle

import settings

from caffevis.caffevis_helper import load_mean
from loaders import load_imagenet_mean, load_labels, caffe
from jby_misc import WithTimer
from max_tracker import scan_images_for_maxes



def main():
    parser = argparse.ArgumentParser(description='Finds images in a training set that cause max activation for a network; saves results in a pickled NetMaxTracker.')
    parser.add_argument('--N', type = int, default = 9, help = 'note and save top N activations')
    parser.add_argument('--gpu', action = 'store_true', default=settings.caffevis_mode_gpu, help = 'use gpu')
    parser.add_argument('--siamese', action = 'store_true', help = 'expected siamese network format')
    parser.add_argument('--net_prototxt', type = str, default = settings.caffevis_deploy_prototxt, help = 'network prototxt to load')
    parser.add_argument('--net_weights', type = str, default = settings.caffevis_network_weights, help = 'network weights to load')
    parser.add_argument('--datadir', type = str, default = settings.static_files_dir, help = 'directory to look for files in')
    parser.add_argument('--filelist', type = str, default = settings.static_files_input_file, help = 'list of image files to consider, one per line')
    parser.add_argument('--outfile', type=str, default = 'find_max_acts_output.pickled', help='output filename for pkl')
    parser.add_argument('--mean', type = str, default = settings.caffevis_data_mean, help = 'data mean to load')
    args = parser.parse_args()

    if args.mean == "":
        mean = load_imagenet_mean()
    else:
        mean = load_mean(args.mean)

    net = caffe.Classifier(args.net_prototxt,
                           args.net_weights,
                           mean=mean,
                           channel_swap=settings.caffe_net_channel_swap,
                           raw_scale=settings.caffe_net_raw_scale,
                           image_dims=settings.caffe_net_image_dims)

    if args.gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    with WithTimer('Scanning images'):
        max_tracker = scan_images_for_maxes(net, args.datadir, args.filelist, args.N)
    with WithTimer('Saving maxes'):
        with open(args.outfile, 'wb') as ff:
            pickle.dump(max_tracker, ff, -1)



if __name__ == '__main__':
    main()
