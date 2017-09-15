import os
import time

import cv2
import numpy as np
import glob
import math

from codependent_thread import CodependentThread
from image_misc import caffe_load_image, ensure_uint255_and_resize_to_fit, cv2_read_file_rgb, \
    ensure_uint255_and_resize_without_fit
from caffevis_helper import crop_to_corner



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


    def load_image_into_pane_original_format(self, state_layer, state_selected_unit, resize_shape, images, sub_folder,
                                             file_pattern, image_index_to_set, should_crop_to_corner=False):

        jpg_path = os.path.join(self.settings.caffevis_unit_jpg_dir, sub_folder, state_layer,
                                file_pattern % (state_layer, state_selected_unit))

        try:
            img = cv2_read_file_rgb(jpg_path)

            if should_crop_to_corner:
                img = crop_to_corner(img, 2)
            images[image_index_to_set] = ensure_uint255_and_resize_without_fit(img, resize_shape)

        except IOError:
            print '\nAttempted to load file %s but failed. To supress this warning, remove layer "%s" from settings.caffevis_jpgvis_layers' % (
            jpg_path, state_layer)
            pass

    def load_image_into_pane_max_tracker_format(self, state_layer, state_selected_unit, resize_shape, images,
                                                file_search_pattern, image_index_to_set, should_crop_to_corner=False):

        unit_folder_path = os.path.join(self.settings.caffevis_unit_jpg_dir, state_layer,
                                        "unit_%04d" % (state_selected_unit),
                                        file_search_pattern)

        try:

            # list unit images
            unit_images_path = sorted(glob.glob(unit_folder_path))

            # load all images
            # unit_images = [caffe_load_image(unit_image_path, color=True) for unit_image_path in unit_images_path]
            unit_images = [cv2_read_file_rgb(unit_image_path) for unit_image_path in unit_images_path]

            if should_crop_to_corner:
                unit_images = [crop_to_corner(img, 2) for img in unit_images]

            # build mega image
            (image_height, image_width, channels) = unit_images[0].shape
            num_images = len(unit_images)
            images_per_axis = int(math.ceil(math.sqrt(num_images)))
            padding_pixel = 1
            mega_image_height = images_per_axis * (image_height + 2*padding_pixel)
            mega_image_width = images_per_axis * (image_width + 2*padding_pixel)
            mega_image = np.zeros((mega_image_height,mega_image_width,channels))

            for i in range(num_images):
                cell_row = i % images_per_axis
                cell_col = i / images_per_axis
                mega_image_height_start = 1 + cell_row * (image_height + 2*padding_pixel)
                mega_image_height_end = mega_image_height_start + image_height
                mega_image_width_start = 1 + cell_col * (image_width + 2*padding_pixel)
                mega_image_width_end = mega_image_width_start + image_width
                mega_image[mega_image_height_start:mega_image_height_end,mega_image_width_start:mega_image_width_end,:] = unit_images[i]

            images[image_index_to_set] = ensure_uint255_and_resize_without_fit(mega_image, resize_shape)

        except:
            print '\nAttempted to load files from %s but failed. To supress this warning, remove layer "%s" from settings.caffevis_jpgvis_layers' % (
                unit_folder_path, state_layer)
            pass


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

            state_layer, state_selected_unit, data_shape = jpgvis_to_load_key

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


            if self.settings.caffevis_unit_jpg_dir_folder_format == 'original_combined_single_image':

                # 0. e.g. regularized_opt/conv1/conv1_0037_montage.jpg
                self.load_image_into_pane_original_format(state_layer, state_selected_unit, resize_shape, images,
                                                          sub_folder='regularized_opt',
                                                          file_pattern='%s_%04d_montage.jpg',
                                                          image_index_to_set=0,
                                                          should_crop_to_corner=True)

            elif self.settings.caffevis_unit_jpg_dir_folder_format == 'max_tracker_output':
                # self.load_image_into_pane_max_tracker_format(state_layer, state_selected_unit, resize_shape, images,
                #                                              file_search_pattern='regularized_opt*.png',
                #                                              image_index_to_set=0,
                #                                              should_crop_to_corner=True)
                # black image until we test it
                images[0] = np.zeros((resize_shape[0],resize_shape[1],3))

            if self.settings.caffevis_unit_jpg_dir_folder_format == 'original_combined_single_image':

                # 1. e.g. max_im/conv1/conv1_0037.jpg
                self.load_image_into_pane_original_format(state_layer, state_selected_unit, resize_shape, images,
                                                          sub_folder='max_im',
                                                          file_pattern='%s_%04d.jpg',
                                                          image_index_to_set=1)

            elif self.settings.caffevis_unit_jpg_dir_folder_format == 'max_tracker_output':
                self.load_image_into_pane_max_tracker_format(state_layer, state_selected_unit, resize_shape, images,
                                                             file_search_pattern='maxim*.png',
                                                             image_index_to_set=1)


            if self.settings.caffevis_unit_jpg_dir_folder_format == 'original_combined_single_image':

                # 2. e.g. max_deconv/conv1/conv1_0037.jpg
                self.load_image_into_pane_original_format(state_layer, state_selected_unit, resize_shape, images,
                                                          sub_folder='max_deconv',
                                                          file_pattern='%s_%04d.jpg',
                                                          image_index_to_set=2)

            elif self.settings.caffevis_unit_jpg_dir_folder_format == 'max_tracker_output':
                self.load_image_into_pane_max_tracker_format(state_layer, state_selected_unit, resize_shape, images,
                                                             file_search_pattern='deconv*.png',
                                                             image_index_to_set=2)

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
