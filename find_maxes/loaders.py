#! /usr/bin/env python

from pylab import *

def load_labels(settings):
    with open('%s/data/ilsvrc12/synset_words.txt' % settings.caffevis_caffe_root) as ff:
        labels = [line.strip() for line in ff.readlines()]
    return labels

    
def load_trained_net(settings, model_prototxt = None, model_weights = None):
    assert (model_prototxt is None) == (model_weights is None), 'Specify both model_prototxt and model_weights or neither'
    if model_prototxt is None:
        load_dir = '/home/jyosinsk/results/140311_234854_afadfd3_priv_netbase_upgraded/'
        model_prototxt = load_dir + 'deploy_1.prototxt'
        model_weights = load_dir + 'caffe_imagenet_train_iter_450000'

    print 'LOADER: loading net:'
    print '  ', model_prototxt
    print '  ', model_weights

    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe
    if settings.caffevis_mode_gpu:
        caffe.set_mode_gpu()
        print 'CaffeVisApp mode (in main thread):     GPU'
    else:
        caffe.set_mode_cpu()
        print 'CaffeVisApp mode (in main thread):     CPU'

    net = caffe.Classifier(model_prototxt, model_weights)
    #net.set_phase_test()

    return net

    
def load_imagenet_mean(settings):
    imagenet_mean = np.load(settings.caffevis_caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    imagenet_mean = imagenet_mean[:, 14:14+227, 14:14+227]    # (3,256,256) -> (3,227,227) Crop to center 227x227 section
    return imagenet_mean
