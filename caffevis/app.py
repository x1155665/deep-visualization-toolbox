#! /usr/bin/env python
# -*- coding: utf-8

# add parent folder to search path, to enable import of core modules like settings
import os,sys,inspect

from dask.array.random import normal

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import cv2
import numpy as np
import StringIO

from find_maxes.find_max_acts import load_max_tracker_from_file
import find_maxes.max_tracker
sys.modules['max_tracker'] = find_maxes.max_tracker

from misc import WithTimer, mkdir_p
from numpy_cache import FIFOLimitedArrayCache
from app_base import BaseApp
from image_misc import norm01, norm01c, tile_images_normalize, ensure_float01, tile_images_make_tiles, \
    ensure_uint255_and_resize_to_fit, resize_without_fit, ensure_uint255, \
    load_image, ensure_uint255_and_resize_without_fit, array_histogram, fig2data
from image_misc import FormattedString, cv2_typeset_text, to_255
from caffe_proc_thread import CaffeProcThread
from caffevis_app_state import CaffeVisAppState, SiameseViewMode, PatternMode, BackpropMode, BackpropViewOption, \
    ColorMapOption, InputOverlayOption
from caffevis_helper import get_pretty_layer_name, read_label_file, load_sprite_image, load_square_sprite_image, \
    get_image_from_files
from caffe_misc import layer_name_to_top_name, save_caffe_image
from siamese_helper import SiameseHelper
from settings_misc import get_receptive_field


class CaffeVisApp(BaseApp):
    '''App to visualize using caffe.'''

    def __init__(self, settings, key_bindings):
        super(CaffeVisApp, self).__init__(settings, key_bindings)

        print 'Got settings', settings
        self.settings = settings
        self.bindings = key_bindings

        self.net, self._data_mean = settings.adapter.load_network(settings)

        # set network batch size to 1
        current_input_shape = self.net.blobs[self.net.inputs[0]].shape
        current_input_shape[0] = 1
        self.net.blobs[self.net.inputs[0]].reshape(*current_input_shape)
        self.net.reshape()

        self._net_channel_swap = settings._calculated_channel_swap

        if self._net_channel_swap is None:
            self._net_channel_swap_inv = None
        else:
            self._net_channel_swap_inv = tuple([self._net_channel_swap.index(ii) for ii in range(len(self._net_channel_swap))])

        self.labels = None
        if self.settings.caffevis_labels:
            self.labels = read_label_file(self.settings.caffevis_labels)
        self.proc_thread = None
        self.jpgvis_thread = None
        self.handled_frames = 0
        if settings.caffevis_jpg_cache_size < 10*1024**2:
            raise Exception('caffevis_jpg_cache_size must be at least 10MB for normal operation.')
        self.img_cache = FIFOLimitedArrayCache(settings.caffevis_jpg_cache_size)

        self.header_boxes = []
        self.buttons_boxes = []

    def start(self, live_vis):
        from jpg_vis_loading_thread import JPGVisLoadingThread

        self.live_vis = live_vis
        self.state = CaffeVisAppState(self.net, self.settings, self.bindings, live_vis)
        self.state.drawing_stale = True
        self.header_print_names = [get_pretty_layer_name(self.settings, nn) for nn in self.state.get_headers()]

        if self.proc_thread is None or not self.proc_thread.is_alive():
            # Start thread if it's not already running
            self.proc_thread = CaffeProcThread(self.settings, self.net, self.state,
                                               self.settings.caffevis_frame_wait_sleep,
                                               self.settings.caffevis_pause_after_keys,
                                               self.settings.caffevis_heartbeat_required,
                                               self.settings.caffevis_mode_gpu)
            self.proc_thread.start()

        if self.jpgvis_thread is None or not self.jpgvis_thread.is_alive():
            # Start thread if it's not already running
            self.jpgvis_thread = JPGVisLoadingThread(self.settings, self.state, self.img_cache,
                                                     self.settings.caffevis_jpg_load_sleep,
                                                     self.settings.caffevis_heartbeat_required)
            self.jpgvis_thread.start()
                

    def get_heartbeats(self):
        return [self.proc_thread.heartbeat, self.jpgvis_thread.heartbeat]
            
    def quit(self):
        print 'CaffeVisApp: trying to quit'

        with self.state.lock:
            self.state.quit = True

        if self.proc_thread != None:
            for ii in range(3):
                self.proc_thread.join(1)
                if not self.proc_thread.is_alive():
                    break
            if self.proc_thread.is_alive():
                raise Exception('CaffeVisApp: Could not join proc_thread; giving up.')
            self.proc_thread = None
                
        print 'CaffeVisApp: quitting.'
        
    def _can_skip_all(self, panes):
        return ('caffevis_layers' not in panes.keys())
        
    def handle_input(self, input_image, input_label, input_filename, panes):
        if self.debug_level > 1:
            print 'handle_input: frame number', self.handled_frames, 'is', 'None' if input_image is None else 'Available'
        self.handled_frames += 1
        if self._can_skip_all(panes):
            return

        with self.state.lock:
            if self.debug_level > 1:
                print 'CaffeVisApp.handle_input: pushed frame'
            self.state.next_frame = input_image
            self.state.next_label = input_label
            self.state.next_filename = input_filename
            if self.debug_level > 1:
                print 'CaffeVisApp.handle_input: caffe_net_state is:', self.state.caffe_net_state

            self.state.last_frame = input_image
    
    def redraw_needed(self):
        return self.state.redraw_needed()

    def draw(self, panes):
        if self._can_skip_all(panes):
            if self.debug_level > 1:
                print 'CaffeVisApp.draw: skipping'
            return False

        with self.state.lock:
            # Hold lock throughout drawing
            do_draw = self.state.drawing_stale and self.state.caffe_net_state == 'free'
            # print 'CaffeProcThread.draw: caffe_net_state is:', self.state.caffe_net_state
            if do_draw:
                self.state.caffe_net_state = 'draw'

        if do_draw:
            if self.debug_level > 1:
                print 'CaffeVisApp.draw: drawing'

            if 'caffevis_control' in panes:
                self._draw_control_pane(panes['caffevis_control'])
            if 'caffevis_status' in panes:
                self._draw_status_pane(panes['caffevis_status'])
            if 'caffevis_buttons' in panes:
                self._draw_buttons_pane(panes['caffevis_buttons'])
            layer_data_3D_highres = None
            if 'caffevis_layers' in panes:
                layer_data_3D_highres = self._draw_layer_pane(panes['caffevis_layers'])
            if 'caffevis_aux' in panes:
                self._draw_aux_pane(panes['caffevis_aux'], layer_data_3D_highres)
            if 'caffevis_back' in panes:
                # Draw back pane as normal
                self._draw_back_pane(panes['caffevis_back'])
                if self.state.layers_pane_zoom_mode == 2:
                    # ALSO draw back pane into layers pane
                    self._draw_back_pane(panes['caffevis_layers'])
            if 'caffevis_jpgvis' in panes:
                self._draw_jpgvis_pane(panes['caffevis_jpgvis'])

            with self.state.lock:
                self.state.drawing_stale = False
                self.state.caffe_net_state = 'free'
        return do_draw

    def _draw_prob_labels_pane(self, pane):
        '''Adds text label annotation atop the given pane.'''

        if not self.labels or not self.state.show_label_predictions or not self.settings.caffevis_prob_layer:
            return

        #pane.data[:] = to_255(self.settings.window_background)
        defaults = {'face':  getattr(cv2, self.settings.caffevis_class_face),
                    'fsize': self.settings.caffevis_class_fsize,
                    'clr':   to_255(self.settings.caffevis_class_clr_0),
                    'thick': self.settings.caffevis_class_thick}
        loc = self.settings.caffevis_class_loc[::-1]   # Reverse to OpenCV c,r order
        clr_0 = to_255(self.settings.caffevis_class_clr_0)
        clr_1 = to_255(self.settings.caffevis_class_clr_1)

        probs_flat = self.net.blobs[layer_name_to_top_name(self.net, self.settings.caffevis_prob_layer)].data.flatten()
        top_5 = probs_flat.argsort()[-1:-6:-1]

        strings = []
        pmax = probs_flat[top_5[0]]
        for idx in top_5:
            prob = probs_flat[idx]
            text = '%.2f %s' % (prob, self.labels[idx])
            fs = FormattedString(text, defaults)
            #fs.clr = tuple([clr_1[ii]*prob/pmax + clr_0[ii]*(1-prob/pmax) for ii in range(3)])
            fs.clr = tuple([max(0,min(255,clr_1[ii]*prob + clr_0[ii]*(1-prob))) for ii in range(3)])
            strings.append([fs])   # Line contains just fs

        cv2_typeset_text(pane.data, strings, loc,
                         line_spacing = self.settings.caffevis_class_line_spacing)

    def _draw_control_pane(self, pane):
        pane.data[:] = to_255(self.settings.window_background)

        with self.state.lock:
            layer_idx = self.state.layer_idx

        loc = self.settings.caffevis_control_loc[::-1]   # Reverse to OpenCV c,r order

        strings = []
        defaults = {'face':  getattr(cv2, self.settings.caffevis_control_face),
                    'fsize': self.settings.caffevis_control_fsize,
                    'clr':   to_255(self.settings.caffevis_control_clr),
                    'thick': self.settings.caffevis_control_thick}

        for ii in range(len(self.header_print_names)):
            fs = FormattedString(self.header_print_names[ii], defaults)
            this_layer_def = self.settings.layers_list[ii]
            if self.state.backprop_selection_frozen and this_layer_def == self.state.get_current_backprop_layer_definition():
                fs.clr = to_255(self.settings.caffevis_control_clr_bp)
                fs.thick = self.settings.caffevis_control_thick_bp
            if this_layer_def == self.state.get_current_layer_definition():
                if self.state.cursor_area == 'top':
                    fs.clr = to_255(self.settings.caffevis_control_clr_cursor)
                    fs.thick = self.settings.caffevis_control_thick_cursor
                else:
                    if not (self.state.backprop_selection_frozen and this_layer_def == self.state.get_current_backprop_layer_definition()):
                        fs.clr = to_255(self.settings.caffevis_control_clr_selected)
                        fs.thick = self.settings.caffevis_control_thick_selected
            strings.append(fs)

        locy, self.header_boxes = cv2_typeset_text(pane.data, strings, loc,
                                                   line_spacing = self.settings.caffevis_control_line_spacing,
                                                   wrap = True)

        if hasattr(self.settings, 'control_pane_height'):
            self.settings._calculated_control_pane_height = self.settings.control_pane_height
        else:
            self.settings._calculated_control_pane_height = locy - loc[1] + 4

    def _draw_status_pane(self, pane):
        pane.data[:] = to_255(self.settings.window_background)

        defaults = {'face':  getattr(cv2, self.settings.caffevis_status_face),
                    'fsize': self.settings.caffevis_status_fsize,
                    'clr':   to_255(self.settings.caffevis_status_clr),
                    'thick': self.settings.caffevis_status_thick}
        loc = self.settings.caffevis_status_loc[::-1]   # Reverse to OpenCV c,r order

        status = StringIO.StringIO()
        status2 = StringIO.StringIO()
        fps = self.proc_thread.approx_fps()
        with self.state.lock:
            pattern_first_mode = "first" if self.state.pattern_first_only else "all"
            if self.state.pattern_mode == PatternMode.MAXIMAL_OPTIMIZED_IMAGE:
                print >> status, 'pattern(' + pattern_first_mode + ' optimized max)'
            elif self.state.pattern_mode == PatternMode.MAXIMAL_INPUT_IMAGE:
                print >> status, 'pattern(' + pattern_first_mode + ' input max)'
            elif self.state.pattern_mode == PatternMode.WEIGHTS_HISTOGRAM:
                print >> status, 'histogram(weights)'
            elif self.state.pattern_mode == PatternMode.MAX_ACTIVATIONS_HISTOGRAM:
                print >> status, 'histogram(maximal activations)'
            elif self.state.pattern_mode == PatternMode.ACTIVATIONS_CORRELATION:
                print >> status, 'correlation(maximal activations)'
            elif self.state.pattern_mode == PatternMode.WEIGHTS_CORRELATION:
                print >> status, 'correlation(weights)'
            elif self.state.layers_show_back:
                print >> status, 'back'
            else:
                print >> status, 'fwd'

            default_layer_name = self.state.get_default_layer_name()
            print >>status, '%s:%d |' % (default_layer_name, self.state.selected_unit),
            if not self.state.back_enabled:
                print >>status, 'Back: off',
            else:
                print >>status, 'Back: %s (%s)' % (BackpropMode.to_string(self.state.back_mode), BackpropViewOption.to_string(self.state.back_view_option)),
                print >>status, '(from %s_%d)' % (self.state.get_default_layer_name(self.state.get_current_backprop_layer_definition()), self.state.backprop_unit),
            print >>status, '|',
            print >>status, 'Boost: %g/%g' % (self.state.layer_boost_indiv, self.state.layer_boost_gamma)

            if fps > 0:
                print >>status, '| FPS: %.01f' % fps

            if self.state.next_label:
                print >> status, '| GT Label: %s' % self.state.next_label

            if self.state.extra_msg:
                print >>status, '|', self.state.extra_msg

            print >> status2, 'Layer size: %s' % (self.state.get_layer_output_size_string())

            print >> status2, '| Receptive field:', '%s' % (str(get_receptive_field(self.settings, self.net, default_layer_name)))

            print >> status2, '| Input: %s' % (str(self.state.next_filename))

        strings_line1 = [FormattedString(line, defaults) for line in status.getvalue().split('\n')]
        strings_line2 = [FormattedString(line, defaults) for line in status2.getvalue().split('\n')]

        locy, boxes = cv2_typeset_text(pane.data, strings_line1, (loc[0], loc[1] + 5),
                                       line_spacing = self.settings.caffevis_status_line_spacing)

        locy, boxes = cv2_typeset_text(pane.data, strings_line2, (loc[0], locy),
                                       line_spacing=self.settings.caffevis_status_line_spacing)

    def _draw_buttons_pane(self, pane):

        pane.data[:] = to_255(self.settings.window_background)

        header_defaults = {'face': getattr(cv2, self.settings.caffevis_buttons_header_face),
                           'fsize': self.settings.caffevis_buttons_header_fsize,
                           'clr': to_255(self.settings.caffevis_buttons_header_clr),
                           'thick': self.settings.caffevis_buttons_header_thick}
        normal_defaults = {'face': getattr(cv2, self.settings.caffevis_buttons_normal_face),
                           'fsize': self.settings.caffevis_buttons_normal_fsize,
                           'clr': to_255(self.settings.caffevis_buttons_normal_clr),
                           'thick': self.settings.caffevis_buttons_normal_thick}
        selected_defaults = {'face': getattr(cv2, self.settings.caffevis_buttons_selected_face),
                           'fsize': self.settings.caffevis_buttons_selected_fsize,
                           'clr': to_255(self.settings.caffevis_buttons_selected_clr),
                           'thick': self.settings.caffevis_buttons_selected_thick}

        loc = self.settings.caffevis_buttons_loc[::-1]  # Reverse to OpenCV c,r order

        text = StringIO.StringIO()
        fps = self.proc_thread.approx_fps()

        lines = list()

        with self.state.lock:
            lines.append([FormattedString('Input', header_defaults)])

            file_defaults = selected_defaults if self.live_vis.input_updater.static_file_mode else normal_defaults
            camera_defaults = selected_defaults if not self.live_vis.input_updater.static_file_mode else normal_defaults

            lines.append([FormattedString('File', file_defaults), FormattedString('Prev', normal_defaults), FormattedString('Next', normal_defaults)])
            lines.append([FormattedString('Camera', camera_defaults)])
            lines.append([FormattedString('', normal_defaults)])

            activations_defaults = selected_defaults if self.state.pattern_mode == PatternMode.OFF and not self.state.layers_show_back else normal_defaults
            gradients_defaults = selected_defaults if self.state.pattern_mode == PatternMode.OFF and self.state.layers_show_back else normal_defaults
            max_optimized_defaults = selected_defaults if self.state.pattern_mode == PatternMode.MAXIMAL_OPTIMIZED_IMAGE else normal_defaults
            max_input_defaults = selected_defaults if self.state.pattern_mode == PatternMode.MAXIMAL_INPUT_IMAGE else normal_defaults
            weights_hist_defaults = selected_defaults if self.state.pattern_mode == PatternMode.WEIGHTS_HISTOGRAM else normal_defaults
            act_hist_defaults = selected_defaults if self.state.pattern_mode == PatternMode.MAX_ACTIVATIONS_HISTOGRAM else normal_defaults
            weights_corr_defaults = selected_defaults if self.state.pattern_mode == PatternMode.WEIGHTS_CORRELATION else normal_defaults
            act_corr_defaults = selected_defaults if self.state.pattern_mode == PatternMode.ACTIVATIONS_CORRELATION else normal_defaults
            lines.append([FormattedString('Modes', header_defaults)])
            lines.append([FormattedString('Activations', activations_defaults)])
            lines.append([FormattedString('Gradients', gradients_defaults)])
            lines.append([FormattedString('Maximal Optimized', max_optimized_defaults)])
            lines.append([FormattedString('Maximal Input', max_input_defaults)])
            lines.append([FormattedString('Weights Histogram', weights_hist_defaults)])
            lines.append([FormattedString('Activations Histogram', act_hist_defaults)])
            lines.append([FormattedString('Weights Correlation', weights_corr_defaults)])
            lines.append([FormattedString('Activations Correlation', act_corr_defaults)])
            lines.append([FormattedString('', normal_defaults)])

            no_overlay_defaults = selected_defaults if self.state.input_overlay_option == InputOverlayOption.OFF else normal_defaults
            over_active_defaults = selected_defaults if self.state.input_overlay_option == InputOverlayOption.OVER_ACTIVE else normal_defaults
            over_inactive_defaults = selected_defaults if self.state.input_overlay_option == InputOverlayOption.OVER_INACTIVE else normal_defaults
            lines.append([FormattedString('Input Overlay', header_defaults)])
            lines.append([FormattedString('No Overlay', no_overlay_defaults)])
            lines.append([FormattedString('Over Active', over_active_defaults)])
            lines.append([FormattedString('Over Inactive', over_inactive_defaults)])
            lines.append([FormattedString('', normal_defaults)])

            backprop_no_defaults = selected_defaults if self.state.back_mode == BackpropMode.OFF else normal_defaults
            backprop_gradients_defaults = selected_defaults if self.state.back_mode == BackpropMode.GRAD else normal_defaults
            backprop_zf_defaults = selected_defaults if self.state.back_mode == BackpropMode.DECONV_ZF else normal_defaults
            backprop_gb_defaults = selected_defaults if self.state.back_mode == BackpropMode.DECONV_GB else normal_defaults
            backprop_frozen_defaults = selected_defaults if self.state.backprop_selection_frozen else normal_defaults
            lines.append([FormattedString('Backprop Modes', header_defaults)])
            lines.append([FormattedString('No Backprop', backprop_no_defaults)])
            lines.append([FormattedString('Gradient', backprop_gradients_defaults)])
            lines.append([FormattedString('ZF Deconv', backprop_zf_defaults)])
            lines.append([FormattedString('Guided Backprop', backprop_gb_defaults)])
            lines.append([FormattedString('Freeze Origin', backprop_frozen_defaults)])
            lines.append([FormattedString('', normal_defaults)])

            backview_raw_defaults = selected_defaults if self.state.back_view_option == BackpropViewOption.RAW else normal_defaults
            backview_gray_defaults = selected_defaults if self.state.back_view_option == BackpropViewOption.GRAY else normal_defaults
            backview_norm_defaults = selected_defaults if self.state.back_view_option == BackpropViewOption.NORM else normal_defaults
            backview_normblur_defaults = selected_defaults if self.state.back_view_option == BackpropViewOption.NORM_BLUR else normal_defaults
            backview_possum_defaults = selected_defaults if self.state.back_view_option == BackpropViewOption.POS_SUM else normal_defaults
            backview_hist_defaults = selected_defaults if self.state.back_view_option == BackpropViewOption.HISTOGRAM else normal_defaults
            lines.append([FormattedString('Backprop Views', header_defaults)])
            lines.append([FormattedString('Raw', backview_raw_defaults)])
            lines.append([FormattedString('Gray', backview_gray_defaults)])
            lines.append([FormattedString('Norm', backview_norm_defaults)])
            lines.append([FormattedString('Blurred Norm', backview_normblur_defaults)])
            lines.append([FormattedString('Sum > 0', backview_possum_defaults)])
            lines.append([FormattedString('Gradient Histogram', backview_hist_defaults)])
            lines.append([FormattedString('', normal_defaults)])

            lines.append([FormattedString('Help', normal_defaults)])
            lines.append([FormattedString('Quit', normal_defaults)])

        # strings_line1 = [[FormattedString(line, defaults)] for line in text.getvalue().split('\n')]

        locy, self.buttons_boxes = cv2_typeset_text(pane.data, lines, (loc[0], loc[1] + 5),
                                       line_spacing=self.settings.caffevis_buttons_line_spacing)

        return


    def prepare_tile_image(self, display_3D, highlight_selected, n_tiles, tile_rows, tile_cols):

        if self.state.layers_show_back and self.state.pattern_mode == PatternMode.OFF:
            padval = self.settings.caffevis_layer_clr_back_background
        else:
            padval = self.settings.window_background

        highlights = [None] * n_tiles
        if highlight_selected:
            with self.state.lock:
                if self.state.cursor_area == 'bottom':
                    highlights[self.state.selected_unit] = self.settings.caffevis_layer_clr_cursor  # in [0,1] range
                if self.state.backprop_selection_frozen and self.state.get_current_layer_definition() == self.state.get_current_backprop_layer_definition():
                    highlights[self.state.backprop_unit] = self.settings.caffevis_layer_clr_back_sel  # in [0,1] range

        _, display_2D = tile_images_make_tiles(display_3D, hw=(tile_rows, tile_cols), padval=padval, highlights=highlights)

        return display_2D

    def _draw_layer_pane(self, pane):
        '''Returns the data shown in highres format, b01c order.'''

        default_layer_name = self.state.get_default_layer_name()

        if self.state.siamese_view_mode_has_two_images():

            if self.state.layers_show_back:

                layer_dat_3D_0, layer_dat_3D_1 = self.state.get_siamese_selected_diff_blobs(self.net)
            else:
                layer_dat_3D_0, layer_dat_3D_1 = self.state.get_siamese_selected_data_blobs(self.net)

            # Promote FC layers with shape (n) to have shape (n,1,1)
            if len(layer_dat_3D_0.shape) == 1:
                layer_dat_3D_0 = layer_dat_3D_0[:, np.newaxis, np.newaxis]
                layer_dat_3D_1 = layer_dat_3D_1[:, np.newaxis, np.newaxis]

                # we don't resize the images to half the size since there is no point in doing that in FC layers
            elif layer_dat_3D_0.shape[2] == 1:
                # we don't resize the images to half the size since it will crash
                pass
            else:
                # resize images to half the size
                half_pane_shape = (layer_dat_3D_0.shape[1], layer_dat_3D_0.shape[2] / 2)
                layer_dat_3D_0 = resize_without_fit(layer_dat_3D_0.transpose((1, 2, 0)), half_pane_shape).transpose((2, 0, 1))
                layer_dat_3D_1 = resize_without_fit(layer_dat_3D_1.transpose((1, 2, 0)), half_pane_shape).transpose((2, 0, 1))

            # concatenate images side-by-side
            layer_dat_3D = np.concatenate((layer_dat_3D_0, layer_dat_3D_1), axis=2)

        else:
            if self.state.layers_show_back:
                layer_dat_3D = self.state.get_single_selected_diff_blob(self.net)
            else:
                layer_dat_3D = self.state.get_single_selected_data_blob(self.net)

        # Promote FC layers with shape (n) to have shape (n,1,1)
        if len(layer_dat_3D.shape) == 1:
            layer_dat_3D = layer_dat_3D[:, np.newaxis, np.newaxis]

        n_tiles = layer_dat_3D.shape[0]

        top_name = layer_name_to_top_name(self.net, default_layer_name)
        tile_rows, tile_cols = self.state.net_blob_info[top_name]['tiles_rc']

        display_2D = None
        display_3D_highres = None
        is_layer_summary_loaded = False
        if self.state.pattern_mode != PatternMode.OFF:
            # Show desired patterns loaded from disk

            if self.state.pattern_mode == PatternMode.MAXIMAL_OPTIMIZED_IMAGE:

                if self.settings.caffevis_outputs_dir_folder_format == 'original_combined_single_image':

                    display_2D, display_3D, display_3D_highres, is_layer_summary_loaded = self.load_pattern_images_original_format(
                        default_layer_name, layer_dat_3D, n_tiles, pane, tile_cols, tile_rows)

                elif self.settings.caffevis_outputs_dir_folder_format == 'max_tracker_output':

                    display_2D, display_3D, display_3D_highres, is_layer_summary_loaded = self.load_pattern_images_optimizer_format(
                        default_layer_name, layer_dat_3D, n_tiles, pane, tile_cols, tile_rows,
                        self.state.pattern_first_only, file_search_pattern='opt*.jpg')

            elif self.state.pattern_mode == PatternMode.MAXIMAL_INPUT_IMAGE:

                if self.settings.caffevis_outputs_dir_folder_format == 'original_combined_single_image':
                    # maximal input image patterns is not implemented in original format
                    display_3D_highres = np.zeros((layer_dat_3D.shape[0], pane.data.shape[0],
                                                   pane.data.shape[1],
                                                   pane.data.shape[2]), dtype=np.uint8)
                    display_3D = self.downsample_display_3d(display_3D_highres, layer_dat_3D, pane, tile_cols, tile_rows)
                    print "ERROR: patterns view with maximal input images is not implemented when settings.caffevis_outputs_dir_folder_format == 'original_combined_single_image'"

                elif self.settings.caffevis_outputs_dir_folder_format == 'max_tracker_output':
                    display_2D, display_3D, display_3D_highres, is_layer_summary_loaded = self.load_pattern_images_optimizer_format(
                        default_layer_name, layer_dat_3D, n_tiles, pane, tile_cols, tile_rows,
                        self.state.pattern_first_only, file_search_pattern='maxim*.png')

            elif self.state.pattern_mode == PatternMode.WEIGHTS_HISTOGRAM:
                display_2D, display_3D, display_3D_highres, is_layer_summary_loaded = self.load_weights_histograms(
                    self.net, default_layer_name, layer_dat_3D, n_tiles, pane, tile_cols, tile_rows,
                    show_layer_summary=self.state.cursor_area == 'top')

            elif self.state.pattern_mode == PatternMode.MAX_ACTIVATIONS_HISTOGRAM:
                if self.settings.caffevis_histograms_format == 'load_from_file':
                    display_2D, display_3D, display_3D_highres, is_layer_summary_loaded = self.load_pattern_images_optimizer_format(
                        default_layer_name, layer_dat_3D, n_tiles, pane, tile_cols, tile_rows, True,
                        file_search_pattern='max_histogram.png',
                        show_layer_summary=self.state.cursor_area == 'top',
                        file_summary_pattern='layer_inactivity.png')

                elif self.settings.caffevis_histograms_format == 'calculate_in_realtime':
                    display_2D, display_3D, display_3D_highres, is_layer_summary_loaded = self.load_maximal_activations_histograms(
                        default_layer_name, layer_dat_3D, n_tiles, pane, tile_cols, tile_rows,
                        show_layer_summary=self.state.cursor_area == 'top')

            elif self.state.pattern_mode == PatternMode.ACTIVATIONS_CORRELATION:
                display_2D, display_3D, display_3D_highres, is_layer_summary_loaded = self.load_pattern_images_optimizer_format(
                    default_layer_name, layer_dat_3D, n_tiles, pane, tile_cols, tile_rows, True,
                    file_search_pattern=None,
                    show_layer_summary=True,
                    file_summary_pattern='channels_correlation.png')

            elif self.state.pattern_mode == PatternMode.WEIGHTS_CORRELATION:
                display_2D, display_3D, display_3D_highres, is_layer_summary_loaded = self.load_weights_correlation(
                    self.net, default_layer_name, layer_dat_3D, n_tiles, pane, tile_cols, tile_rows,
                    show_layer_summary=True)

            else:
                raise Exception("Invalid value of pattern mode: %d" % self.state.pattern_mode)
        else:

            # Show data from network (activations or diffs)
            if self.state.layers_show_back:
                back_what_to_disp = self.get_back_what_to_disp()
                if back_what_to_disp == 'disabled':
                    layer_dat_3D_normalized = np.tile(self.settings.window_background, layer_dat_3D.shape + (1,))
                elif back_what_to_disp == 'stale':
                    layer_dat_3D_normalized = np.tile(self.settings.stale_background, layer_dat_3D.shape + (1,))
                else:
                    layer_dat_3D_normalized = tile_images_normalize(layer_dat_3D,
                                                                    boost_indiv = self.state.layer_boost_indiv,
                                                                    boost_gamma = self.state.layer_boost_gamma,
                                                                    neg_pos_colors = ((1,0,0), (0,1,0)))
            else:
                layer_dat_3D_normalized = tile_images_normalize(layer_dat_3D,
                                                                boost_indiv = self.state.layer_boost_indiv,
                                                                boost_gamma = self.state.layer_boost_gamma)

            display_3D = layer_dat_3D_normalized

        # Convert to float if necessary:
        display_3D = ensure_float01(display_3D)
        # Upsample gray -> color if necessary
        #   e.g. (1000,32,32) -> (1000,32,32,3)
        if len(display_3D.shape) == 3:
            display_3D = display_3D[:,:,:,np.newaxis]
        if display_3D.shape[3] == 1:
            display_3D = np.tile(display_3D, (1, 1, 1, 3))
        # Upsample unit length tiles to give a more sane tile / highlight ratio
        #   e.g. (1000,1,1,3) -> (1000,3,3,3)
        if (display_3D.shape[1] == 1) and (display_3D.shape[2] == 1):
            display_3D = np.tile(display_3D, (1, 3, 3, 1))
        # Upsample pair of unit length tiles to give a more sane tile / highlight ratio (occurs on siamese FC layers)
        #   e.g. (1000,1,2,3) -> (1000,2,2,3)
        if (display_3D.shape[1] == 1) and (display_3D.shape[2] == 2):
            display_3D = np.tile(display_3D, (1, 2, 1, 1))

        if display_3D_highres is None:
            display_3D_highres = display_3D


        # generate 2D display by tiling the 3D images and add highlights, unless already generated
        if display_2D is None:
            display_2D = self.prepare_tile_image(display_3D, True, n_tiles, tile_rows, tile_cols)

        self._display_pane_based_on_zoom_mode(display_2D, display_3D_highres, is_layer_summary_loaded, pane)

        self._add_label_or_score_overlay(default_layer_name, pane)
            
        return display_3D_highres

    def _display_pane_based_on_zoom_mode(self, display_2D, display_3D_highres, is_layer_summary_loaded, pane):
        # Display pane based on layers_pane_zoom_mode
        state_layers_pane_zoom_mode = self.state.layers_pane_zoom_mode
        assert state_layers_pane_zoom_mode in (0, 1, 2)
        if state_layers_pane_zoom_mode == 0:
            # Mode 0: normal display (activations or patterns)
            if self.settings.caffevis_keep_aspect_ratio:
                display_2D_resize = ensure_uint255_and_resize_to_fit(display_2D, pane.data.shape)
            else:
                display_2D_resize = ensure_uint255_and_resize_without_fit(display_2D, pane.data.shape)

        elif state_layers_pane_zoom_mode == 1 and not is_layer_summary_loaded:
            # Mode 1: zoomed selection
            display_2D_resize = self.get_processed_selected_unit(display_3D_highres, pane.data.shape, use_colored_data=False)

        elif state_layers_pane_zoom_mode == 2 and not is_layer_summary_loaded:
            # Mode 2: zoomed backprop pane
            if self.settings.caffevis_keep_aspect_ratio:
                display_2D_resize = ensure_uint255_and_resize_to_fit(display_2D, pane.data.shape) * 0
            else:
                display_2D_resize = ensure_uint255_and_resize_without_fit(display_2D, pane.data.shape) * 0

        else:  # any other case = zoom_mode + is_layer_summary_loaded
            if self.settings.caffevis_keep_aspect_ratio:
                display_2D_resize = ensure_uint255_and_resize_to_fit(display_2D, pane.data.shape)
            else:
                display_2D_resize = ensure_uint255_and_resize_without_fit(display_2D, pane.data.shape)
        pane.data[:] = to_255(self.settings.window_background)
        pane.data[0:display_2D_resize.shape[0], 0:display_2D_resize.shape[1], :] = display_2D_resize

    def _add_label_or_score_overlay(self, default_layer_name, pane):

        if self.state.cursor_area == 'bottom':

            # Display label annotation atop layers pane (e.g. for fc8/prob)
            defaults = {'face': getattr(cv2, self.settings.caffevis_label_face),
                        'fsize': self.settings.caffevis_label_fsize,
                        'clr': to_255(self.settings.caffevis_label_clr),
                        'thick': self.settings.caffevis_label_thick}
            loc_base = self.settings.caffevis_label_loc[::-1]  # Reverse to OpenCV c,r order

            text_to_display = ""
            if (self.labels) and (default_layer_name in self.settings.caffevis_label_layers):
                text_to_display = self.labels[self.state.selected_unit] + " "

            if self.state.show_maximal_score:
                if self.state.siamese_view_mode_has_two_images():
                    if self.state.layers_show_back:
                        blob1, blob2 = self.state.get_siamese_selected_diff_blobs(self.net)

                        if len(blob1.shape) == 1:
                            value1, value2 = blob1[self.state.selected_unit], blob2[self.state.selected_unit]
                            text_to_display += 'grad: ' + str(value1) + " " + str(value2)
                    else:
                        blob1, blob2 = self.state.get_siamese_selected_data_blobs(self.net)

                        if len(blob1.shape) == 1:
                            value1, value2 = blob1[self.state.selected_unit], blob2[self.state.selected_unit]
                            text_to_display += 'act: ' + str(value1) + " " + str(value2)

                else:
                    if self.state.layers_show_back:
                        blob = self.state.get_single_selected_diff_blob(self.net)

                        if len(blob.shape) == 1:
                            value = blob[self.state.selected_unit]
                            text_to_display += 'grad: ' + str(value)
                    else:
                        blob = self.state.get_single_selected_data_blob(self.net)

                        if len(blob.shape) == 1:
                            value = blob[self.state.selected_unit]
                            text_to_display += 'act: ' + str(value)

            lines = [FormattedString(text_to_display, defaults)]
            cv2_typeset_text(pane.data, lines, loc_base)

    def load_pattern_images_original_format(self, default_layer_name, layer_dat_3D, n_tiles, pane,
                                            tile_cols, tile_rows):
        display_2D = None
        display_3D_highres = None
        is_layer_summary_loaded = False
        load_layer = default_layer_name
        if self.settings.caffevis_jpgvis_remap and load_layer in self.settings.caffevis_jpgvis_remap:
            load_layer = self.settings.caffevis_jpgvis_remap[load_layer]
        if ((self.settings.caffevis_jpgvis_layers and load_layer in self.settings.caffevis_jpgvis_layers) or (self.settings.caffevis_jpgvis_layers is None)) and self.settings.caffevis_outputs_dir:
            jpg_path = os.path.join(self.settings.caffevis_outputs_dir, 'regularized_opt', load_layer, 'whole_layer.jpg')

            # Get highres version
            display_3D_highres = self.img_cache.get((jpg_path, 'whole'), None)

            if display_3D_highres is None:
                try:
                    with WithTimer('CaffeVisApp:load_sprite_image', quiet=self.debug_level < 1):
                        display_3D_highres = load_square_sprite_image(jpg_path, n_sprites=n_tiles)
                except IOError:
                    # File does not exist, so just display disabled.
                    pass
                else:
                    if display_3D_highres is not None:
                        self.img_cache.set((jpg_path, 'whole'), display_3D_highres)

        display_3D = self.downsample_display_3d(display_3D_highres, layer_dat_3D, pane, tile_cols, tile_rows)
        return display_2D, display_3D, display_3D_highres, is_layer_summary_loaded

    def load_pattern_images_optimizer_format(self, default_layer_name, layer_dat_3D, n_tiles, pane,
                                            tile_cols, tile_rows, first_only, file_search_pattern, show_layer_summary = False, file_summary_pattern = ""):
        is_layer_summary_loaded = False
        display_2D = None
        display_3D_highres = None
        load_layer = default_layer_name
        if self.settings.caffevis_jpgvis_remap and load_layer in self.settings.caffevis_jpgvis_remap:
            load_layer = self.settings.caffevis_jpgvis_remap[load_layer]
        if (self.settings.caffevis_jpgvis_layers and load_layer in self.settings.caffevis_jpgvis_layers) or (self.settings.caffevis_jpgvis_layers is None):

            # get number of units
            units_num = layer_dat_3D.shape[0]

            pattern_image_key = (self.settings.caffevis_outputs_dir, load_layer, "unit_%04d", units_num, file_search_pattern, first_only, show_layer_summary, file_summary_pattern)

            # Get highres version
            display_3D_highres = self.img_cache.get(pattern_image_key, None)

            if display_3D_highres is None:
                try:

                    if self.settings.caffevis_outputs_dir:
                        resize_shape = pane.data.shape

                        if show_layer_summary:
                            # load layer summary image
                            layer_summary_image_path = os.path.join(self.settings.caffevis_outputs_dir, load_layer, file_summary_pattern)
                            layer_summary_image = load_image(layer_summary_image_path, color=True, as_uint=True)
                            layer_summary_image = ensure_uint255_and_resize_without_fit(layer_summary_image, resize_shape)
                            display_3D_highres = layer_summary_image
                            display_3D_highres = np.expand_dims(display_3D_highres, 0)
                            display_2D = display_3D_highres[0]
                            is_layer_summary_loaded = True

                        else:
                            if file_search_pattern is None:
                                display_3D_highres = None
                            else:
                                with WithTimer('CaffeVisApp:load_image_per_unit', quiet=self.debug_level < 1):
                                    # load all images
                                    display_3D_highres = self.load_image_per_unit(display_3D_highres, load_layer, units_num, first_only, resize_shape, file_search_pattern)

                except IOError:
                    # File does not exist, so just display disabled.
                    pass
                else:
                    if display_3D_highres is not None:
                        self.img_cache.set(pattern_image_key, display_3D_highres)
            else:

                # if layer found in cache, mark it as loaded
                if show_layer_summary:
                    display_2D = display_3D_highres[0]
                    is_layer_summary_loaded = True

        display_3D = self.downsample_display_3d(display_3D_highres, layer_dat_3D, pane, tile_cols, tile_rows)
        return display_2D, display_3D, display_3D_highres, is_layer_summary_loaded

    def load_image_per_unit(self, display_3D_highres, load_layer, units_num, first_only, resize_shape, file_search_pattern):

        # limit loading
        if units_num > 1000:
            print "WARNING: load_image_per_unit was asked to load %d units, aborted to avoid hang" % (units_num)
            return None

        # for each neuron in layer
        for unit_id in range(0, units_num):

            unit_folder_path = os.path.join(self.settings.caffevis_outputs_dir, load_layer, "unit_%04d" % (unit_id), file_search_pattern)

            try:
                if unit_id % 10 == 0:
                    print "loading %s images for layer %s channel %d out of %d" % (file_search_pattern, load_layer, unit_id, units_num)

                unit_first_image = get_image_from_files(self.settings, unit_folder_path, False, resize_shape, first_only)

                # handle first generation of results container
                if display_3D_highres is None:
                    unit_first_image_shape = unit_first_image.shape
                    display_3D_highres = np.zeros((units_num, unit_first_image_shape[0],
                                                   unit_first_image_shape[1],
                                                   unit_first_image_shape[2]), dtype=np.uint8)

                # set in result
                display_3D_highres[unit_id, :, ::] = unit_first_image

            except:
                print '\nAttempted to load file from %s but failed. To supress this warning, remove layer "%s" from settings.caffevis_jpgvis_layers' % \
                      (unit_folder_path, load_layer)
                pass
        return display_3D_highres

    def downsample_display_3d(self, display_3D_highres, layer_dat_3D, pane, tile_cols, tile_rows):
        if display_3D_highres is not None:
            # Get lowres version, maybe. Assume we want at least one pixel for selection border.
            row_downsamp_factor = int(
                np.ceil(float(display_3D_highres.shape[1]) / (pane.data.shape[0] / tile_rows - 2)))
            col_downsamp_factor = int(
                np.ceil(float(display_3D_highres.shape[2]) / (pane.data.shape[1] / tile_cols - 2)))
            ds = max(row_downsamp_factor, col_downsamp_factor)
            if ds > 1:
                # print 'Downsampling by', ds
                display_3D = display_3D_highres[:, ::ds, ::ds, :]
            else:
                display_3D = display_3D_highres
        else:
            display_3D = layer_dat_3D * 0  # nothing to show
        return display_3D


    def load_weights_histograms(self, net, layer_name, layer_dat_3D, n_channels, pane, tile_cols, tile_rows, show_layer_summary):

        is_layer_summary_loaded = False
        display_2D = None
        display_3D = None
        empty_display_3D = np.zeros(layer_dat_3D.shape + (3,))

        pattern_image_key_3d = (layer_name, "weights_histogram", show_layer_summary, self.state.selected_unit, "3D")
        pattern_image_key_2d = (layer_name, "weights_histogram", show_layer_summary, self.state.selected_unit, "2D")

        # Get highres version
        display_3D_highres = self.img_cache.get(pattern_image_key_3d, None)
        display_2D = self.img_cache.get(pattern_image_key_2d, None)

        if display_3D_highres is None or display_2D is None:

            pane_shape = pane.data.shape

            if not self.settings.caffevis_outputs_dir:
                folder_path = None
                cache_layer_weights_histogram_image_path = None
                cache_details_weights_histogram_image_path = None
            else:
                folder_path = os.path.join(self.settings.caffevis_outputs_dir, layer_name)
                cache_layer_weights_histogram_image_path = os.path.join(folder_path, 'layer_weights_histogram.png')
                cache_details_weights_histogram_image_path = os.path.join(folder_path, 'details_weights_histogram.png')

            # plotting objects needed for
            # 1. calculating size of results array
            # 2. generating weights histogram for selected unit
            # 3. generating weights histograms for all units

            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(10, 10), facecolor='white', tight_layout=False)
            ax = fig.add_subplot(111)

            def calculate_weights_histogram_for_specific_unit(channel_idx, fig, ax, do_print):

                if do_print and channel_idx % 10 == 0:
                    print "calculating weights histogram for layer %s channel %d out of %d" % (layer_name, channel_idx, n_channels)

                # get vector of weights
                weights = net.params[layer_name][0].data[channel_idx].flatten()
                bias = net.params[layer_name][1].data[channel_idx]

                # create histogram
                hist, bin_edges = np.histogram(weights, bins=50)

                # generate histogram image file
                width = 0.7 * (bin_edges[1] - bin_edges[0])
                center = (bin_edges[:-1] + bin_edges[1:]) / 2

                ax.bar(center, hist, align='center', width=width, color='g')

                fig.suptitle('weights for unit %d, bias is %f' % (channel_idx, bias))
                ax.xaxis.label.set_text('weight value')
                ax.yaxis.label.set_text('count')

                figure_buffer = fig2data(fig)

                display_3D_highres[channel_idx, :, ::] = figure_buffer

                ax.cla()


            try:

                # handle generation of results container
                figure_buffer = fig2data(fig)
                first_shape = figure_buffer.shape
                display_3D_highres = np.zeros((n_channels, first_shape[0], first_shape[1], first_shape[2]), dtype=np.uint8)

                # try load from cache
                if show_layer_summary:

                    # try load cache file for layer weight histogram
                    if cache_layer_weights_histogram_image_path and os.path.exists(cache_layer_weights_histogram_image_path):

                        # load 2d image from cache file
                        display_2D = load_image(cache_layer_weights_histogram_image_path, color=True, as_uint=False)
                        display_3D_highres = np.zeros(pane_shape)
                        display_3D_highres = np.expand_dims(display_3D_highres, 0)
                        display_3D_highres[0] = display_2D

                        is_layer_summary_loaded = True

                else:

                    # try load cache file for details weights histogram
                    if cache_details_weights_histogram_image_path and os.path.exists(cache_details_weights_histogram_image_path):

                        # load 2d image from cache file
                        display_2D = load_image(cache_details_weights_histogram_image_path, color=True, as_uint=False)

                        # calculate weights histogram for selected unit
                        calculate_weights_histogram_for_specific_unit(self.state.selected_unit, fig, ax, do_print=False)

                        display_3D = self.downsample_display_3d(display_3D_highres, layer_dat_3D, pane, tile_cols, tile_rows)

                        # generate empty highlights
                        display_2D_highlights_only = self.prepare_tile_image(display_3D * 0, True, n_channels, tile_rows, tile_cols)

                        # if shapes are not equal, cache is invalid
                        if display_2D_highlights_only.shape == display_2D.shape:
                            # mix highlights with cached image
                            display_2D = (display_2D_highlights_only != 0) * display_2D_highlights_only + (display_2D_highlights_only == 0) * display_2D
                        else:
                            display_2D = None

                # if not loaded from cache, generate the data
                if display_2D is None:

                    # calculate weights histogram image

                    # check if layer has weights at all
                    if not net.params.has_key(layer_name):
                        return display_2D, empty_display_3D, empty_display_3D, is_layer_summary_loaded

                    # pattern_image_key_layer = (layer_name, "weights_histogram", True)
                    # pattern_image_key_details = (layer_name, "weights_histogram", False)

                    # self.img_cache.set(pattern_image_key_details, display_3D_highres)
                    # self.img_cache.set(pattern_image_key_layer, display_3D_highres_summary)

                    if show_layer_summary:

                        half_pane_shape = (pane_shape[0], pane_shape[1] / 2)

                        # generate weights histogram for layer
                        weights = net.params[layer_name][0].data.flatten()
                        hist, bin_edges = np.histogram(weights, bins=50)

                        width = 0.7 * (bin_edges[1] - bin_edges[0])
                        center = (bin_edges[:-1] + bin_edges[1:]) / 2
                        ax.bar(center, hist, align='center', width=width, color='g')

                        fig.suptitle('weights for layer %s' % layer_name)
                        ax.xaxis.label.set_text('weight value')
                        ax.yaxis.label.set_text('count')

                        figure_buffer = fig2data(fig)
                        display_3D_highres_summary_weights = ensure_uint255_and_resize_without_fit(figure_buffer, half_pane_shape)

                        ax.cla()

                        # generate bias histogram for layer
                        bias = net.params[layer_name][1].data.flatten()
                        hist, bin_edges = np.histogram(bias, bins=50)

                        width = 0.7 * (bin_edges[1] - bin_edges[0])
                        center = (bin_edges[:-1] + bin_edges[1:]) / 2
                        ax.bar(center, hist, align='center', width=width, color='g')

                        fig.suptitle('bias for layer %s' % layer_name)
                        ax.xaxis.label.set_text('bias value')
                        ax.yaxis.label.set_text('count')

                        figure_buffer = fig2data(fig)
                        display_3D_highres_summary_bias = ensure_uint255_and_resize_without_fit(figure_buffer, half_pane_shape)

                        display_3D_highres_summary = np.concatenate((display_3D_highres_summary_weights, display_3D_highres_summary_bias), axis=1)
                        display_3D_highres_summary = np.expand_dims(display_3D_highres_summary, 0)

                        display_3D_highres = display_3D_highres_summary
                        display_2D = display_3D_highres[0]
                        is_layer_summary_loaded = True

                        if folder_path:
                            mkdir_p(folder_path)
                            save_caffe_image(display_2D[:,:,::-1].astype(np.float32).transpose((2,0,1)), cache_layer_weights_histogram_image_path)
                        else:
                            print "WARNING: unable to save weight histogram to cache since caffevis_outputs_dir is not set"

                    else:

                        # for each channel
                        for channel_idx in xrange(n_channels):
                            calculate_weights_histogram_for_specific_unit(channel_idx, fig, ax, do_print=True)

                        display_3D = self.downsample_display_3d(display_3D_highres, layer_dat_3D, pane, tile_cols, tile_rows)

                        # generate display of details weights histogram image
                        display_2D = self.prepare_tile_image(display_3D, False, n_channels, tile_rows, tile_cols)

                        if folder_path:
                            # save histogram image to cache
                            mkdir_p(folder_path)
                            save_caffe_image(display_2D[:,:,::-1].astype(np.float32).transpose((2,0,1)), cache_details_weights_histogram_image_path)
                        else:
                            print "WARNING: unable to save weight histogram to cache since caffevis_outputs_dir is not set"

                        # generate empty highlights
                        display_2D_highlights_only = self.prepare_tile_image(display_3D * 0, True, n_channels, tile_rows, tile_cols)

                        # mix highlights with cached image
                        display_2D = (display_2D_highlights_only != 0) * display_2D_highlights_only + (display_2D_highlights_only == 0) * display_2D

            except IOError:
                return display_2D, empty_display_3D, empty_display_3D, is_layer_summary_loaded
                # File does not exist, so just display disabled.
                pass

            else:
                self.img_cache.set(pattern_image_key_3d, display_3D_highres)
                self.img_cache.set(pattern_image_key_2d, display_2D)

            fig.clf()
            plt.close(fig)

        else:
            # here we can safely assume that display_2D is not None, so we only need to check if show_layer_summary was requested
            if show_layer_summary:
                is_layer_summary_loaded = True

            pass

        if display_3D is None:
            display_3D = self.downsample_display_3d(display_3D_highres, layer_dat_3D, pane, tile_cols, tile_rows)

        return display_2D, display_3D, display_3D_highres, is_layer_summary_loaded

    def load_weights_correlation(self, net, layer_name, layer_dat_3D, n_channels, pane, tile_cols, tile_rows, show_layer_summary):

        is_layer_summary_loaded = False
        display_2D = None
        display_3D = None
        empty_display_3D = np.zeros(layer_dat_3D.shape + (3,))

        pattern_image_key_3d = (layer_name, "weights_correlation", show_layer_summary, self.state.selected_unit, "3D")
        pattern_image_key_2d = (layer_name, "weights_correlation", show_layer_summary, self.state.selected_unit, "2D")

        # Get highres version
        display_3D_highres = self.img_cache.get(pattern_image_key_3d, None)
        display_2D = self.img_cache.get(pattern_image_key_2d, None)

        if display_3D_highres is None or display_2D is None:

            pane_shape = pane.data.shape

            if not self.settings.caffevis_outputs_dir:
                folder_path = None
                cache_layer_weights_correlation_image_path = None
            else:
                folder_path = os.path.join(self.settings.caffevis_outputs_dir, layer_name)
                cache_layer_weights_correlation_image_path = os.path.join(folder_path, 'layer_weights_correlation.png')

            try:

                # try load cache file for layer weight correlation
                if cache_layer_weights_correlation_image_path and os.path.exists(cache_layer_weights_correlation_image_path):

                    # load 2d image from cache file
                    display_2D = load_image(cache_layer_weights_correlation_image_path, color=True, as_uint=False)
                    display_3D_highres = np.zeros(pane_shape)
                    display_3D_highres = np.expand_dims(display_3D_highres, 0)
                    display_3D_highres[0] = display_2D

                    is_layer_summary_loaded = True

                # if not loaded from cache, generate the data
                if display_2D is None:

                    # calculate weights correlation image

                    # check if layer has weights at all
                    if not net.params.has_key(layer_name):
                        return display_2D, empty_display_3D, empty_display_3D, is_layer_summary_loaded

                    # skip layers with only one channel
                    if n_channels == 1:
                        return display_2D, empty_display_3D, empty_display_3D, is_layer_summary_loaded

                    data_unroll = net.params[layer_name][0].data.reshape((n_channels, -1))  # Note: no copy eg (96,3025). Does nothing if not is_spatial

                    corr = np.corrcoef(data_unroll)

                    # fix possible NANs
                    corr = np.nan_to_num(corr)
                    np.fill_diagonal(corr, 1)

                    # sort correlation matrix
                    indexes = np.lexsort(corr)
                    sorted_corr = corr[indexes, :][:, indexes]

                    # plot correlation matrix
                    import matplotlib.pyplot as plt

                    fig = plt.figure(figsize=(10, 10), facecolor='white', tight_layout=True)
                    plt.subplot(1, 1, 1)
                    plt.imshow(sorted_corr, interpolation='nearest', vmin=-1, vmax=1)
                    plt.colorbar()
                    plt.title('channels weights correlation matrix for layer %s' % (layer_name))
                    figure_buffer = fig2data(fig)
                    plt.close()

                    display_3D_highres_summary = ensure_uint255_and_resize_without_fit(figure_buffer, pane_shape)
                    display_3D_highres_summary = np.expand_dims(display_3D_highres_summary, 0)
                    display_3D_highres = display_3D_highres_summary
                    display_2D = display_3D_highres[0]
                    is_layer_summary_loaded = True

                    if folder_path:
                        mkdir_p(folder_path)
                        save_caffe_image(display_2D[:,:,::-1].astype(np.float32).transpose((2,0,1)), cache_layer_weights_correlation_image_path)
                    else:
                        print "WARNING: unable to save weight correlationto cache since caffevis_outputs_dir is not set"

                self.img_cache.set(pattern_image_key_3d, display_3D_highres)
                self.img_cache.set(pattern_image_key_2d, display_2D)

            except IOError:
                return display_2D, empty_display_3D, empty_display_3D, is_layer_summary_loaded
                # File does not exist, so just display disabled.
                pass

        else:
            # here we can safely assume that display_2D is not None, so we only need to check if show_layer_summary was requested
            if show_layer_summary:
                is_layer_summary_loaded = True

            pass

        if display_3D is None:
            display_3D = self.downsample_display_3d(display_3D_highres, layer_dat_3D, pane, tile_cols, tile_rows)

        return display_2D, display_3D, display_3D_highres, is_layer_summary_loaded

    def load_maximal_activations_histograms(self, default_layer_name, layer_dat_3D, n_tiles, pane, tile_cols, tile_rows, show_layer_summary):

        display_2D = None
        empty_display_3D = np.zeros(layer_dat_3D.shape + (3,))

        is_layer_summary_loaded = False

        maximum_activation_histogram_data_file = os.path.join(settings.caffevis_outputs_dir, 'find_max_acts_output.pickled')
        pattern_image_key = (maximum_activation_histogram_data_file, default_layer_name, "max histograms", show_layer_summary)

        # Get highres version
        display_3D_highres = self.img_cache.get(pattern_image_key, None)

        pane_shape = pane.data.shape

        if display_3D_highres is None:
            try:
                # load pickle file
                net_max_tracker = load_max_tracker_from_file(maximum_activation_histogram_data_file)

                if not net_max_tracker.max_trackers.has_key(default_layer_name):
                    return display_2D, empty_display_3D, empty_display_3D, is_layer_summary_loaded

                # check if
                if not hasattr(net_max_tracker.max_trackers[default_layer_name], 'channel_to_histogram'):
                    print "ERROR: file %s is missing the field channel_to_histogram, try rerun find_max_acts to generate it" % (maximum_activation_histogram_data_file)
                    return display_2D, empty_display_3D, empty_display_3D, is_layer_summary_loaded

                channel_to_histogram = net_max_tracker.max_trackers[default_layer_name].channel_to_histogram

                def channel_to_histogram_values(channel_idx):

                    # get channel data
                    hist, bin_edges = channel_to_histogram[channel_idx]

                    return hist, bin_edges

                display_3D_highres_list = [display_3D_highres, display_3D_highres]

                def process_channel_figure(channel_idx, fig):
                    figure_buffer = fig2data(fig)

                    # handle first generation of results container
                    if display_3D_highres_list[0] is None:
                        first_shape = figure_buffer.shape
                        display_3D_highres_list[0] = np.zeros((len(channel_to_histogram), first_shape[0],
                                                       first_shape[1],
                                                       first_shape[2]), dtype=np.uint8)

                    display_3D_highres_list[0][channel_idx, :, ::] = figure_buffer
                    pass

                def process_layer_figure(fig):
                    figure_buffer = fig2data(fig)
                    display_3D_highres_list[1] = ensure_uint255_and_resize_without_fit(figure_buffer, pane_shape)
                    display_3D_highres_list[1] = np.expand_dims(display_3D_highres_list[1], 0)
                    pass

                n_channels = len(channel_to_histogram)
                find_maxes.max_tracker.prepare_max_histogram(default_layer_name, n_channels, channel_to_histogram_values, process_channel_figure, process_layer_figure)

                pattern_image_key_layer = (maximum_activation_histogram_data_file, default_layer_name, "max histograms",True)
                pattern_image_key_details = (maximum_activation_histogram_data_file, default_layer_name, "max histograms",False)

                self.img_cache.set(pattern_image_key_details, display_3D_highres_list[0])
                self.img_cache.set(pattern_image_key_layer, display_3D_highres_list[1])

                if show_layer_summary:
                    display_3D_highres = display_3D_highres_list[1]
                    display_2D = display_3D_highres[0]
                    is_layer_summary_loaded = True
                else:
                    display_3D_highres = display_3D_highres_list[0]

            except IOError:
                return display_2D, empty_display_3D, empty_display_3D, is_layer_summary_loaded
                # File does not exist, so just display disabled.
                pass
        else:

            # if layer found in cache, mark it as loaded
            if show_layer_summary:
                display_2D = display_3D_highres[0]
                is_layer_summary_loaded = True

        display_3D = self.downsample_display_3d(display_3D_highres, layer_dat_3D, pane, tile_cols, tile_rows)
        return display_2D, display_3D, display_3D_highres, is_layer_summary_loaded

    def get_processed_selected_unit(self, all_units, resize_shape, use_colored_data = False):

        unit_data = all_units[self.state.selected_unit]
        if self.settings.caffevis_keep_aspect_ratio:
            unit_data_resize = resize_to_fit(unit_data, resize_shape)
        else:
            unit_data_resize = resize_without_fit(unit_data, resize_shape)

        if self.state.pattern_mode == PatternMode.OFF:
            if self.state.last_frame is None:
                pass

            input_image = SiameseHelper.get_image_from_frame(self.state.last_frame, self.state.settings.is_siamese,
                                                             resize_shape, self.state.siamese_view_mode)
            normalized_mask = unit_data_resize

            if use_colored_data:
                unit_data_resize = self.state.gray_to_colormap(unit_data_resize)
                normalized_mask = np.tile(normalized_mask[:, :, np.newaxis], 3)

            if self.state.input_overlay_option == InputOverlayOption.OFF:
                pass

            elif self.state.input_overlay_option == InputOverlayOption.OVER_ACTIVE:

                unit_data_resize = normalized_mask * input_image + (1 - normalized_mask) * unit_data_resize

            elif self.state.input_overlay_option == InputOverlayOption.OVER_INACTIVE:
                unit_data_resize = (normalized_mask < 0.1) * input_image + (normalized_mask >= 0.1) * unit_data_resize
                pass

        unit_data_resize = ensure_uint255(unit_data_resize)
        return unit_data_resize


    def _mix_input_overlay_with_colormap_old(self, unit_data, resize_shape, input_image):

        if self.settings.caffevis_keep_aspect_ratio:
            unit_data_resize = ensure_uint255_and_resize_to_fit(unit_data, resize_shape)
            input_image_resize = ensure_uint255_and_resize_to_fit(input_image, resize_shape)
        else:
            unit_data_resize = ensure_uint255_and_resize_without_fit(unit_data, resize_shape)
            input_image_resize = ensure_uint255_and_resize_without_fit(input_image, resize_shape)

        normalized_mask = unit_data_resize / 255.0
        normalized_mask = np.tile(normalized_mask[:, :, np.newaxis], 3)

        colored_unit_data_resize = self.state.gray_to_colormap(unit_data_resize)
        colored_unit_data_resize = ensure_uint255(colored_unit_data_resize)
        if len(colored_unit_data_resize.shape) == 2:
            colored_unit_data_resize = np.tile(colored_unit_data_resize[:, :, np.newaxis], 3)

        if self.state.input_overlay_option == InputOverlayOption.OFF:
            pass

        elif self.state.input_overlay_option == InputOverlayOption.OVER_ACTIVE:
            colored_unit_data_resize = np.array(normalized_mask * input_image_resize + (1 - normalized_mask) * colored_unit_data_resize, dtype = 'uint8')

        elif self.state.input_overlay_option == InputOverlayOption.OVER_INACTIVE:
            MAGIC_THRESHOLD_NUMBER = 0.3
            colored_unit_data_resize = (normalized_mask < MAGIC_THRESHOLD_NUMBER) * input_image_resize + (normalized_mask >= MAGIC_THRESHOLD_NUMBER) * colored_unit_data_resize
            pass

        return colored_unit_data_resize

    def _mix_input_overlay_with_colormap(self, unit_data, resize_shape, input_image):

        # resize
        if self.settings.caffevis_keep_aspect_ratio:
            input_image_resize = resize_to_fit(input_image, resize_shape)
            unit_data_resize = resize_to_fit(unit_data, resize_shape)
        else:
            input_image_resize = resize_without_fit(input_image, resize_shape)
            unit_data_resize = resize_without_fit(unit_data, resize_shape)

        sigma = 0.02 * max(unit_data_resize.shape[0:2])
        blur_unit_data_resize = cv2.GaussianBlur(unit_data_resize, (0, 0), sigma)
        normalized_blur_unit_data_resize = norm01(blur_unit_data_resize)

        colored_normalized_blur_unit_data_resize = self.state.gray_to_colormap(normalized_blur_unit_data_resize)
        if len(colored_normalized_blur_unit_data_resize.shape) == 2:
            colored_normalized_blur_unit_data_resize = np.tile(colored_normalized_blur_unit_data_resize[:, :, np.newaxis], 3)

        if self.state.input_overlay_option == InputOverlayOption.OFF:
            attMap = colored_normalized_blur_unit_data_resize
            pass

        elif self.state.input_overlay_option == InputOverlayOption.OVER_ACTIVE:
            MAGIC_NUMBER = 0.8
            boost_normalized_blur_unit_data_resize = normalized_blur_unit_data_resize ** MAGIC_NUMBER
            boost_normalized_blur_unit_data_resize = boost_normalized_blur_unit_data_resize.reshape(boost_normalized_blur_unit_data_resize.shape + (1,))
            attMap = (boost_normalized_blur_unit_data_resize) * input_image_resize + (1 - boost_normalized_blur_unit_data_resize) * colored_normalized_blur_unit_data_resize

        elif self.state.input_overlay_option == InputOverlayOption.OVER_INACTIVE:
            MAGIC_NUMBER = 0.8
            boost_normalized_blur_unit_data_resize = normalized_blur_unit_data_resize ** MAGIC_NUMBER
            boost_normalized_blur_unit_data_resize = boost_normalized_blur_unit_data_resize.reshape(boost_normalized_blur_unit_data_resize.shape + (1,))
            attMap = (1 - boost_normalized_blur_unit_data_resize) * input_image_resize + (boost_normalized_blur_unit_data_resize) * colored_normalized_blur_unit_data_resize

        return attMap


    def _draw_aux_pane(self, pane, layer_data_normalized):
        pane.data[:] = to_255(self.settings.window_background)

        mode = None
        with self.state.lock:
            if self.state.cursor_area == 'bottom':
                mode = 'selected'
            else:
                mode = 'prob_labels'
                
        if mode == 'selected':
            unit_data_resize = self.get_processed_selected_unit(layer_data_normalized, pane.data.shape, use_colored_data=False)
            pane.data[0:unit_data_resize.shape[0], 0:unit_data_resize.shape[1], :] = unit_data_resize

        elif mode == 'prob_labels':
            self._draw_prob_labels_pane(pane)

    def _draw_back_pane(self, pane):
        mode = None
        with self.state.lock:
            back_enabled = self.state.back_enabled
            back_mode = self.state.back_mode
            back_view_option = self.state.back_view_option
            back_what_to_disp = self.get_back_what_to_disp()
                
        if back_what_to_disp == 'disabled':
            pane.data[:] = to_255(self.settings.window_background)

        elif back_what_to_disp == 'stale':
            pane.data[:] = to_255(self.settings.stale_background)

        else: # One of the backprop modes is enabled and the back computation (gradient or deconv) is up to date

            # define helper function to run processing once or twice, in case of siamese network
            def run_processing_once_or_twice(resize_shape, process_image_fn):

                has_pair_inputs = False
                no_spatial_info = False;

                # if selection is frozen we use the currently selected layer as target for visualization
                if self.state.backprop_selection_frozen:
                    if self.state.siamese_view_mode_has_two_images():
                        grad_blob1, grad_blob2 = self.state.get_siamese_selected_diff_blobs(self.net)

                        if len(grad_blob1.shape) == 1:
                            no_spatial_info = True

                        if len(grad_blob1.shape) == 3:
                            grad_blob1 = grad_blob1.transpose((1, 2, 0))  # c01 -> 01c
                            grad_blob2 = grad_blob2.transpose((1, 2, 0))  # c01 -> 01c

                        has_pair_inputs = True

                    else:
                        grad_blob = self.state.get_single_selected_diff_blob(self.net)
                        if len(grad_blob.shape) == 1:
                            no_spatial_info = True
                        if len(grad_blob.shape) == 3:
                            grad_blob = grad_blob.transpose((1, 2, 0))  # c01 -> 01c

                # if selection is not frozen we use the input layer as target for visualization
                if (not self.state.backprop_selection_frozen) or no_spatial_info:
                    grad_blob = self.net.blobs['data'].diff

                    grad_blob = grad_blob[0]  # bc01 -> c01
                    grad_blob = grad_blob.transpose((1, 2, 0))  # c01 -> 01c

                    if self._net_channel_swap_inv:
                        grad_blob = grad_blob[:, :, self._net_channel_swap_inv]  # e.g. BGR -> RGB

                    # split image to image0 and image1
                    if self.settings.is_siamese:
                        # split image to image0 and image1
                        if self.settings.siamese_input_mode == 'concat_channelwise':
                            [grad_blob1, grad_blob2] = np.split(grad_blob, 2, axis=2)

                        elif self.settings.siamese_input_mode == 'concat_along_width':
                            half_width = grad_blob.shape[1] / 2
                            grad_blob1 = grad_blob[:, :half_width, :]
                            grad_blob2 = grad_blob[:, half_width:, :]

                        has_pair_inputs = True

                # if siamese network, run processing twice
                if self.settings.is_siamese:

                    # combine image0 and image1
                    if self.state.siamese_view_mode == SiameseViewMode.FIRST_IMAGE:
                        # run processing on image0
                        return process_image_fn(grad_blob, resize_shape, self.state.last_frame[0])

                    elif self.state.siamese_view_mode == SiameseViewMode.SECOND_IMAGE:
                        # run processing on image1
                        return process_image_fn(grad_blob, resize_shape, self.state.last_frame[1])

                    elif self.state.siamese_view_mode == SiameseViewMode.BOTH_IMAGES and has_pair_inputs:

                        # resize each gradient image to half the pane size
                        half_pane_shape = (resize_shape[0], resize_shape[1] / 2)

                        # run processing on both image0 and image1
                        image1 = process_image_fn(grad_blob1, half_pane_shape, self.state.last_frame[0])
                        image2 = process_image_fn(grad_blob2, half_pane_shape, self.state.last_frame[1])

                        image1 = resize_without_fit(image1[:], half_pane_shape)
                        image2 = resize_without_fit(image2[:], half_pane_shape)

                        # generate the pane image by concatenating both images
                        return np.concatenate((image1, image2), axis=1)
                    elif self.state.siamese_view_mode == SiameseViewMode.BOTH_IMAGES and not has_pair_inputs:
                        processed_input = self.state.convert_image_pair_to_network_input_format(self.settings, self.state.last_frame, resize_shape)
                        return process_image_fn(grad_blob, resize_shape, processed_input)

                    else:
                        return process_image_fn(grad_blob, resize_shape, self.state.last_frame)

                # else, normal network, run processing once
                else:
                    # run processing on image
                    return process_image_fn(grad_blob, resize_shape, self.state.last_frame)

                raise Exception("flow should not arrive here")

            if back_view_option == BackpropViewOption.RAW:
                def do_raw(grad_blob, resize_shape, input_image):
                    if len(grad_blob.shape) == 3 and grad_blob.shape[2] != 3:
                        return np.zeros(resize_shape)
                    return norm01c(grad_blob, 0)
                grad_img = run_processing_once_or_twice(pane.data.shape, do_raw)

            elif back_view_option == BackpropViewOption.GRAY:
                def do_gray(grad_blob, resize_shape, input_image):
                    return norm01c(grad_blob.mean(axis=2), 0)
                grad_img = run_processing_once_or_twice(pane.data.shape, do_gray)

            elif back_view_option == BackpropViewOption.NORM:
                def do_norm(grad_blob, resize_shape, input_image):
                    norm_grad_blob = norm01(np.linalg.norm(grad_blob, axis=2))
                    return self._mix_input_overlay_with_colormap(norm_grad_blob, resize_shape, input_image)
                grad_img = run_processing_once_or_twice(pane.data.shape, do_norm)

            elif back_view_option == BackpropViewOption.NORM_BLUR:
                def do_norm_blur(grad_blob, resize_shape, input_image):
                    grad_blob = np.linalg.norm(grad_blob, axis=2)
                    cv2.GaussianBlur(grad_blob, (0, 0), self.settings.caffevis_grad_norm_blur_radius, grad_blob)
                    norm_grad_blob = norm01(grad_blob)
                    return self._mix_input_overlay_with_colormap(norm_grad_blob, resize_shape, input_image)
                grad_img = run_processing_once_or_twice(pane.data.shape, do_norm_blur)

            elif back_view_option == BackpropViewOption.POS_SUM:
                def do_pos_sum(grad_blob, resize_shape, input_image):
                    grad_blob = np.maximum(grad_blob.sum(-1), 0)
                    norm_grad_blob = norm01(grad_blob)
                    return self._mix_input_overlay_with_colormap(norm_grad_blob, resize_shape, input_image)
                grad_img = run_processing_once_or_twice(pane.data.shape, do_pos_sum)

            elif back_view_option == BackpropViewOption.HISTOGRAM:
                def do_histogram(grad_blob, resize_shape, input_image):
                    return array_histogram(grad_blob, half_pane_shape, BackpropMode.to_string(back_mode)+' histogram', 'values', 'count')

                half_pane_shape = (pane.data.shape[0],pane.data.shape[1]/2,3)
                grad_img = run_processing_once_or_twice(pane.data.shape, do_histogram)

            else:
                raise Exception('Invalid option for back_view_option: %s' % (back_view_option))

            # If necessary, re-promote from grayscale to color
            if len(grad_img.shape) == 2:
                grad_img = np.tile(grad_img[:,:,np.newaxis], 3)

            if self.settings.caffevis_keep_aspect_ratio:
                grad_img_resize = ensure_uint255_and_resize_to_fit(grad_img, pane.data.shape)
            else:
                grad_img_resize = ensure_uint255_and_resize_without_fit(grad_img, pane.data.shape)

            pane.data[0:grad_img_resize.shape[0], 0:grad_img_resize.shape[1], :] = grad_img_resize

    def _draw_jpgvis_pane(self, pane):
        pane.data[:] = to_255(self.settings.window_background)

        with self.state.lock:
            state_layer_name, state_selected_unit, cursor_area, show_unit_jpgs = self.state.get_default_layer_name(), self.state.selected_unit, self.state.cursor_area, self.state.show_unit_jpgs

        if self.settings.caffevis_jpgvis_remap and state_layer_name in self.settings.caffevis_jpgvis_remap:
            img_key_layer = self.settings.caffevis_jpgvis_remap[state_layer_name]
        else:
            img_key_layer = state_layer_name

        if ((self.settings.caffevis_jpgvis_layers and img_key_layer in self.settings.caffevis_jpgvis_layers) or (self.settings.caffevis_jpgvis_layers is None)) and \
                        cursor_area == 'bottom' and show_unit_jpgs:
            img_key = (img_key_layer, state_selected_unit, pane.data.shape, self.state.show_maximal_score)
            img_resize = self.img_cache.get(img_key, None)
            if img_resize is None:
                # If img_resize is None, loading has not yet been attempted, so show stale image and request load by JPGVisLoadingThread
                with self.state.lock:
                    self.state.jpgvis_to_load_key = img_key
                pane.data[:] = to_255(self.settings.stale_background)
            elif img_resize.nbytes == 0:
                # This is the sentinal value when the image is not
                # found, i.e. loading was already attempted but no jpg
                # assets were found. Just display disabled.
                pane.data[:] = to_255(self.settings.window_background)
            else:
                # Show image
                pane.data[:img_resize.shape[0], :img_resize.shape[1], :] = img_resize
        else:
            # Will never be available
            pane.data[:] = to_255(self.settings.window_background)

    def handle_key(self, key, panes):
        return self.state.handle_key(key)

    def handle_mouse_left_click(self, x, y, flags, param, panes):
        self.state.handle_mouse_left_click(x, y, flags, param, panes, self.header_boxes, self.buttons_boxes)

    def get_back_what_to_disp(self):
        '''Whether to show back diff information or stale or disabled indicator'''
        if (self.state.cursor_area == 'top' and not self.state.backprop_selection_frozen) or not self.state.back_enabled:
            return 'disabled'
        elif self.state.back_stale:
            return 'stale'
        else:
            return 'normal'

    def set_debug(self, level):
        self.debug_level = level
        self.proc_thread.debug_level = level
        self.jpgvis_thread.debug_level = level

    def draw_help(self, help_pane, locy):
        defaults = {'face':  getattr(cv2, self.settings.help_face),
                    'fsize': self.settings.help_fsize,
                    'clr':   to_255(self.settings.help_clr),
                    'thick': self.settings.help_thick}
        loc_base = self.settings.help_loc[::-1]   # Reverse to OpenCV c,r order
        locx = loc_base[0]

        lines = []
        lines.append([FormattedString('DeepVis keys', defaults)])
        
        kl,_ = self.bindings.get_key_help('sel_left')
        kr,_ = self.bindings.get_key_help('sel_right')
        ku,_ = self.bindings.get_key_help('sel_up')
        kd,_ = self.bindings.get_key_help('sel_down')
        klf,_ = self.bindings.get_key_help('sel_left_fast')
        krf,_ = self.bindings.get_key_help('sel_right_fast')
        kuf,_ = self.bindings.get_key_help('sel_up_fast')
        kdf,_ = self.bindings.get_key_help('sel_down_fast')

        keys_nav_0 = ','.join([kk[0] for kk in (kl, kr, ku, kd)])
        keys_nav_1 = ''
        if len(kl)>1 and len(kr)>1 and len(ku)>1 and len(kd)>1:
            keys_nav_1 += ' or '
            keys_nav_1 += ','.join([kk[1] for kk in (kl, kr, ku, kd)])
        keys_nav_f = ','.join([kk[0] for kk in (klf, krf, kuf, kdf)])
        nav_string = 'Navigate with %s%s. Use %s to move faster.' % (keys_nav_0, keys_nav_1, keys_nav_f)
        lines.append([FormattedString('', defaults, width=120, align='right'),
                      FormattedString(nav_string, defaults)])

        for tag in ('help_mode', 'static_file_increment', 'static_file_decrement', 'sel_layer_left', 'sel_layer_right',
                    '', 'next_pattern_mode', 'pattern_first_only', '', 'next_input_overlay', 'next_ez_back_mode_loop',
                    'next_back_view_option', 'next_color_map', '', 'freeze_back_unit', 'show_back', 'zoom_mode',
                    'siamese_view_mode', 'toggle_maximal_score', 'boost_gamma', 'boost_individual', 'freeze_cam',
                    'toggle_input_mode', 'stretch_mode', '', 'reset_state', 'quit'):

            if (tag == ''):
                lines.append([FormattedString('', defaults)])

            else:
                key_strings, help_string = self.bindings.get_key_help(tag)
                label = '%10s:' % (','.join(key_strings))
                lines.append([FormattedString(label, defaults, width=120, align='right'),
                              FormattedString(help_string, defaults)])

        locy, boxes = cv2_typeset_text(help_pane.data, lines, (locx, locy),
                                       line_spacing = self.settings.help_line_spacing)

        return locy
