#! /usr/bin/env python

import os
import time
import errno
import re


class WithTimer:
    def __init__(self, title = '', quiet = False):
        self.title = title
        self.quiet = quiet
        
    def elapsed(self):
        return time.time() - self.wall, time.clock() - self.proc

    def enter(self):
        '''Manually trigger enter'''
        self.__enter__()
    
    def __enter__(self):
        self.proc = time.clock()
        self.wall = time.time()
        return self
        
    def __exit__(self, *args):
        if not self.quiet:
            titlestr = (' ' + self.title) if self.title else ''
            print 'Elapsed%s: wall: %.06f, sys: %.06f' % ((titlestr,) + self.elapsed())



def mkdir_p(path):
    # From https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



def combine_dicts(dicts_tuple):
    '''Combines multiple dictionaries into one by adding a prefix to keys'''
    ret = {}
    for prefix,dictionary in dicts_tuple:
        for key in dictionary.keys():
            ret['%s%s' % (prefix, key)] = dictionary[key]
    return ret


def tsplit(string, no_empty_strings, *delimiters):
    # split string using multiple delimiters

    pattern = '|'.join(map(re.escape, delimiters))
    strings = re.split(pattern, string)
    if no_empty_strings:
        strings = filter(None, strings)

    return strings


def get_files_from_directory(settings):
    # returns list of files in requested directory

    available_files = []
    match_flags = re.IGNORECASE if settings.static_files_ignore_case else 0
    for filename in os.listdir(settings.static_files_dir):
        if re.match(settings.static_files_regexp, filename, match_flags):
            available_files.append(filename)

    return available_files


def get_files_from_image_list(settings):
    # returns list of files in requested image list file

    available_files = []
    labels = []

    with open(settings.static_files_input_file, 'r') as image_list_file:
        lines = image_list_file.readlines()
        # take first token from each line
        available_files = [tsplit(line, True, ' ', ',', '\t')[0] for line in lines if line.strip() != ""]
        labels = [tsplit(line, True, ' ', ',', '\t')[1] for line in lines if line.strip() != ""]

    return available_files, labels


def get_files_from_siamese_image_list(settings):
    # returns list of pair files in requested siamese image list file

    available_files = []
    labels = []

    with open(settings.static_files_input_file, 'r') as image_list_file:
        lines = image_list_file.readlines()
        # take first and second tokens from each line
        available_files = [(tsplit(line, True, ' ', ',', '\t')[0], tsplit(line, True, ' ', ',', '\t')[1])
                           for line in lines if line.strip() != ""]
        labels = [tsplit(line, True, ' ', ',', '\t')[2] for line in lines if line.strip() != ""]

    return available_files, labels


def get_files_list(settings, should_convert_labels = False):

    # available_files - local list of files
    if settings.static_files_input_mode == "directory":
        available_files = get_files_from_directory(settings)
        labels = None
    elif (settings.static_files_input_mode == "image_list") and (not settings.is_siamese):
        available_files, labels = get_files_from_image_list(settings)
    elif (settings.static_files_input_mode == "image_list") and (settings.is_siamese):
        available_files, labels = get_files_from_siamese_image_list(settings)
    else:
        raise Exception(('Error: setting static_files_input_mode has invalid option (%s)' %
                         (settings.static_files_input_mode)))

    if should_convert_labels and labels:
        labels = [settings.convert_label_fn(label) for label in labels]

    return available_files, labels
