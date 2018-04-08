import os
import time

import cv2
import numpy as np
import glob
import math

from codependent_thread import CodependentThread
from image_misc import caffe_load_image, ensure_uint255_and_resize_to_fit, \
    ensure_uint255_and_resize_without_fit
from caffevis_helper import crop_to_corner, get_image_from_files

import caffe

class JPGVisLoadingThread(CodependentThread):
    '''Loads JPGs necessary for caffevis_jpgvis pane in separate
    thread and inserts them into the cache.
    '''

    def __init__(self, settings, state, cache, loop_sleep, heartbeat_required):
        CodependentThread.__init__(self, heartbeat_required)
        self.daemon = True
        self.settings = settings
        self.state = state
        self.cache = cache
        self.loop_sleep = loop_sleep
        self.debug_level = 0


    def load_image_into_pane_original_format(self, state_layer_name, state_selected_unit, resize_shape, images, sub_folder,
                                             file_pattern, image_index_to_set, should_crop_to_corner=False):

        jpg_path = os.path.join(self.settings.caffevis_outputs_dir, sub_folder, state_layer_name,
                                file_pattern % (state_layer_name, state_selected_unit))

        try:
            img = caffe.io.load_image(jpg_path)

            if should_crop_to_corner:
                img = crop_to_corner(img, 2)
            images[image_index_to_set] = ensure_uint255_and_resize_without_fit(img, resize_shape)

        except IOError:
            print '\nAttempted to load file %s but failed. To supress this warning, remove layer "%s" from settings.caffevis_jpgvis_layers' % \
                  (jpg_path, state_layer_name)
            # set black image as place holder
            images[image_index_to_set] = np.zeros((resize_shape[0], resize_shape[1], 3), dtype=np.uint8)
            pass

    def get_score_values_for_max_input_images(self, state_layer_name, state_selected_unit):

        try:

            info_file_path = os.path.join(self.settings.caffevis_outputs_dir, state_layer_name,
                                            "unit_%04d" % (state_selected_unit),
                                            "info.txt")

            # open file
            with open(info_file_path, 'r') as info_file:
                lines = info_file.readlines()

                # skip first line
                lines = lines[1:]

                # take second word from each line, and convert to float
                values = [float(line.split(' ')[1]) for line in lines]

            return values

        except IOError:
            return []
            pass

    def load_image_into_pane_max_tracker_format(self, state_layer_name, state_selected_unit, resize_shape, images,
                                                file_search_pattern, image_index_to_set, should_crop_to_corner=False, first_only = False, captions = [], values = []):

        unit_folder_path = os.path.join(self.settings.caffevis_outputs_dir, state_layer_name,
                                        "unit_%04d" % (state_selected_unit),
                                        file_search_pattern)

        images[image_index_to_set] = get_image_from_files(self.settings, unit_folder_path, should_crop_to_corner, resize_shape, first_only, captions, values)
        return

    def run(self):
        print 'JPGVisLoadingThread.run called'
        
        while not self.is_timed_out():
            with self.state.lock:
                if self.state.quit:
                    break

                #print 'JPGVisLoadingThread.run: caffe_net_state is:', self.state.caffe_net_state
                #print 'JPGVisLoadingThread.run loop: next_frame: %s, caffe_net_state: %s, back_enabled: %s' % (
                #    'None' if self.state.next_frame is None else 'Avail',
                #    self.state.caffe_net_state,
                #    self.state.back_enabled)

                jpgvis_to_load_key = self.state.jpgvis_to_load_key

            if jpgvis_to_load_key is None:
                time.sleep(self.loop_sleep)
                continue

            state_layer_name, state_selected_unit, data_shape, show_maximal_score = jpgvis_to_load_key

            # Load three images:
            images = [None] * 3

            # Resize each component images only using one direction as
            # a constraint. This is straightforward but could be very
            # wasteful (making an image much larger then much smaller)
            # if the proportions of the stacked image are very
            # different from the proportions of the data pane.
            #resize_shape = (None, data_shape[1]) if self.settings.caffevis_jpgvis_stack_vert else (data_shape[0], None)
            # As a heuristic, instead just assume the three images are of the same shape.
            if self.settings.caffevis_jpgvis_stack_vert:
                resize_shape = (data_shape[0]/3, data_shape[1])
            else:
                resize_shape = (data_shape[0], data_shape[1]/3)


            if self.settings.caffevis_outputs_dir_folder_format == 'original_combined_single_image':

                # 0. e.g. regularized_opt/conv1/conv1_0037_montage.jpg
                self.load_image_into_pane_original_format(state_layer_name, state_selected_unit, resize_shape, images,
                                                          sub_folder='regularized_opt',
                                                          file_pattern='%s_%04d_montage.jpg',
                                                          image_index_to_set=0,
                                                          should_crop_to_corner=True)

            elif self.settings.caffevis_outputs_dir_folder_format == 'max_tracker_output':
                self.load_image_into_pane_max_tracker_format(state_layer_name, state_selected_unit, resize_shape, images,
                                                             file_search_pattern='opt*.jpg',
                                                             image_index_to_set=0)

            if self.settings.caffevis_outputs_dir_folder_format == 'original_combined_single_image':

                # 1. e.g. max_im/conv1/conv1_0037.jpg
                self.load_image_into_pane_original_format(state_layer_name, state_selected_unit, resize_shape, images,
                                                          sub_folder='max_im',
                                                          file_pattern='%s_%04d.jpg',
                                                          image_index_to_set=1)

            elif self.settings.caffevis_outputs_dir_folder_format == 'max_tracker_output':

                # convert to string with 2 decimal digits
                values = self.get_score_values_for_max_input_images(state_layer_name, state_selected_unit)

                if self.state.show_maximal_score:
                    captions = [('%.2f' % value) for value in values]
                else:
                    captions = []
                self.load_image_into_pane_max_tracker_format(state_layer_name, state_selected_unit, resize_shape, images,
                                                             file_search_pattern='maxim*.png',
                                                             image_index_to_set=1, captions=captions, values=values)


            if self.settings.caffevis_outputs_dir_folder_format == 'original_combined_single_image':

                # 2. e.g. max_deconv/conv1/conv1_0037.jpg
                self.load_image_into_pane_original_format(state_layer_name, state_selected_unit, resize_shape, images,
                                                          sub_folder='max_deconv',
                                                          file_pattern='%s_%04d.jpg',
                                                          image_index_to_set=2)

            elif self.settings.caffevis_outputs_dir_folder_format == 'max_tracker_output':
                # convert to string with 2 decimal digits
                values = self.get_score_values_for_max_input_images(state_layer_name, state_selected_unit)

                self.load_image_into_pane_max_tracker_format(state_layer_name, state_selected_unit, resize_shape, images,
                                                             file_search_pattern='deconv*.png',
                                                             image_index_to_set=2, values=values)

            # Prune images that were not found:
            images = [im for im in images if im is not None]
            
            # Stack together
            if len(images) > 0:
                #print 'Stacking:', [im.shape for im in images]
                stack_axis = 0 if self.settings.caffevis_jpgvis_stack_vert else 1
                img_stacked = np.concatenate(images, axis = stack_axis)
                #print 'Stacked:', img_stacked.shape
                img_resize = ensure_uint255_and_resize_to_fit(img_stacked, data_shape)
                #print 'Resized:', img_resize.shape
            else:
                img_resize = np.zeros(shape=(0,))   # Sentinal value when image is not found.
                
            self.cache.set(jpgvis_to_load_key, img_resize)

            with self.state.lock:
                self.state.jpgvis_to_load_key = None
                self.state.drawing_stale = True

        print 'JPGVisLoadingThread.run: finished'
