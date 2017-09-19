import os
import time
from threading import Lock
from siamese_helper import SiameseInputMode, SiameseHelper
from caffe_misc import layer_name_to_top_name
from image_misc import get_tiles_height_width_ratio

class PatternMode:
    OFF = 0
    MAXIMAL_OPTIMIZED_IMAGE = 1
    MAXIMAL_INPUT_IMAGE = 2
    MAX_ACTIVATIONS_HISTOGRAM = 3
    NUMBER_OF_MODES = 4

class CaffeVisAppState(object):
    '''State of CaffeVis app.'''

    def __init__(self, net, settings, bindings):
        self.lock = Lock()  # State is accessed in multiple threads
        self.settings = settings
        self.bindings = bindings
        self.net = net

        self.fill_layers_list(net)

        self.siamese_helper = SiameseHelper(settings.layers_list)

        self._populate_net_blob_info(net)

        self.layer_boost_indiv_choices = self.settings.caffevis_boost_indiv_choices   # 0-1, 0 is noop
        self.layer_boost_gamma_choices = self.settings.caffevis_boost_gamma_choices   # 0-inf, 1 is noop
        self.caffe_net_state = 'free'     # 'free', 'proc', or 'draw'
        self.extra_msg = ''
        self.back_stale = True       # back becomes stale whenever the last back diffs were not computed using the current backprop unit and method (bprop or deconv)
        self.next_frame = None
        self.next_label = None
        self.jpgvis_to_load_key = None
        self.last_key_at = 0
        self.quit = False

        self._reset_user_state()

    def _populate_net_blob_info(self, net):
        '''For each blob, save the number of filters and precompute
        tile arrangement (needed by CaffeVisAppState to handle keyboard navigation).
        '''
        self.net_blob_info = {}
        for key in net.blobs.keys():
            self.net_blob_info[key] = {}
            # Conv example: (1, 96, 55, 55)
            # FC example: (1, 1000)
            blob_shape = net.blobs[key].data.shape
            assert len(blob_shape) in (2,4), 'Expected either 2 for FC or 4 for conv layer'
            self.net_blob_info[key]['isconv'] = (len(blob_shape) == 4)
            self.net_blob_info[key]['data_shape'] = blob_shape[1:]  # Chop off batch size
            self.net_blob_info[key]['n_tiles'] = blob_shape[1]
            self.net_blob_info[key]['tiles_rc'] = get_tiles_height_width_ratio(blob_shape[1], self.settings.caffevis_layers_aspect_ratio)
            self.net_blob_info[key]['tile_rows'] = self.net_blob_info[key]['tiles_rc'][0]
            self.net_blob_info[key]['tile_cols'] = self.net_blob_info[key]['tiles_rc'][1]

    def get_headers(self):

        headers = list()
        for layer_def in self.settings.layers_list:
            headers.append(SiameseHelper.get_header_from_layer_def(layer_def))

        return headers

    def _reset_user_state(self):
        self.siamese_input_mode = SiameseInputMode.BOTH_IMAGES
        self.show_maximal_score = False
        self.layer_idx = 0
        self.layer_boost_indiv_idx = self.settings.caffevis_boost_indiv_default_idx
        self.layer_boost_indiv = self.layer_boost_indiv_choices[self.layer_boost_indiv_idx]
        self.layer_boost_gamma_idx = self.settings.caffevis_boost_gamma_default_idx
        self.layer_boost_gamma = self.layer_boost_gamma_choices[self.layer_boost_gamma_idx]
        self.cursor_area = 'top'   # 'top' or 'bottom'
        self.selected_unit = 0
        # Which layer and unit (or channel) to use for backprop
        self.backprop_layer_idx = self.layer_idx
        self.backprop_unit = self.selected_unit
        self.backprop_selection_frozen = False    # If false, backprop unit tracks selected unit
        self.back_enabled = False
        self.back_mode = 'grad'      # 'grad' or 'deconv'
        self.back_filt_mode = 'raw'  # 'raw', 'gray', 'norm', 'normblur'
        self.pattern_mode = PatternMode.OFF    # type of patterns to show instead of activations in layers pane: maximal optimized image, maximal input image, maximal histogram, off
        self.pattern_first_only = True         # should we load only the first pattern image for each neuron, or all the relevant images per neuron
        self.layers_pane_zoom_mode = 0       # 0: off, 1: zoom selected (and show pref in small pane), 2: zoom backprop
        self.layers_show_back = False   # False: show forward activations. True: show backward diffs
        self.show_label_predictions = self.settings.caffevis_init_show_label_predictions
        self.show_unit_jpgs = self.settings.caffevis_init_show_unit_jpgs
        self.drawing_stale = True
        kh,_ = self.bindings.get_key_help('help_mode')
        self.extra_msg = '%s for help' % kh[0]

    def handle_key(self, key):
        #print 'Ignoring key:', key
        if key == -1:
            return key

        with self.lock:
            key_handled = True
            self.last_key_at = time.time()
            tag = self.bindings.get_tag(key)
            if tag == 'reset_state':
                self._reset_user_state()
            elif tag == 'sel_layer_left':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = max(0, self.layer_idx - 1)

            elif tag == 'sel_layer_right':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = min(len(self.settings.layers_list) - 1, self.layer_idx + 1)

            elif tag == 'sel_left':
                self.move_selection('left')
            elif tag == 'sel_right':
                self.move_selection('right')
            elif tag == 'sel_down':
                self.move_selection('down')
            elif tag == 'sel_up':
                self.move_selection('up')

            elif tag == 'sel_left_fast':
                self.move_selection('left', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_right_fast':
                self.move_selection('right', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_down_fast':
                self.move_selection('down', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_up_fast':
                self.move_selection('up', self.settings.caffevis_fast_move_dist)

            elif tag == 'boost_individual':
                self.layer_boost_indiv_idx = (self.layer_boost_indiv_idx + 1) % len(self.layer_boost_indiv_choices)
                self.layer_boost_indiv = self.layer_boost_indiv_choices[self.layer_boost_indiv_idx]
            elif tag == 'boost_gamma':
                self.layer_boost_gamma_idx = (self.layer_boost_gamma_idx + 1) % len(self.layer_boost_gamma_choices)
                self.layer_boost_gamma = self.layer_boost_gamma_choices[self.layer_boost_gamma_idx]
            elif tag == 'next_pattern_mode':
                self.pattern_mode = (self.pattern_mode + 1) % PatternMode.NUMBER_OF_MODES
            elif tag == 'prev_pattern_mode':
                self.pattern_mode = (self.pattern_mode - 1 + PatternMode.NUMBER_OF_MODES) % PatternMode.NUMBER_OF_MODES
            elif tag == 'pattern_first_only':
                self.pattern_first_only = not self.pattern_first_only
            elif tag == 'show_back':
                # If in pattern mode: switch to fwd/back. Else toggle fwd/back mode
                if self.pattern_mode != PatternMode.OFF:
                    self.pattern_mode = PatternMode.OFF
                else:
                    self.layers_show_back = not self.layers_show_back
                if self.layers_show_back:
                    if not self.back_enabled:
                        self.back_enabled = True
                        self.back_stale = True
            elif tag == 'back_mode':
                if not self.back_enabled:
                    self.back_enabled = True
                    self.back_mode = 'grad'
                    self.back_stale = True
                else:
                    if self.back_mode == 'grad':
                        self.back_mode = 'deconv'
                        self.back_stale = True
                    else:
                        self.back_enabled = False
            elif tag == 'back_filt_mode':
                    if self.back_filt_mode == 'raw':
                        self.back_filt_mode = 'gray'
                    elif self.back_filt_mode == 'gray':
                        self.back_filt_mode = 'norm'
                    elif self.back_filt_mode == 'norm':
                        self.back_filt_mode = 'normblur'
                    else:
                        self.back_filt_mode = 'raw'
            elif tag == 'next_ez_back_mode_loop':
                # Cycle:
                # off -> grad (raw) -> grad(gray) -> grad(norm) -> grad(normblur) -> deconv
                if not self.back_enabled:
                    self.back_enabled = True
                    self.back_mode = 'grad'
                    self.back_filt_mode = 'raw'
                    self.back_stale = True
                elif self.back_mode == 'grad' and self.back_filt_mode == 'raw':
                    self.back_filt_mode = 'norm'
                elif self.back_mode == 'grad' and self.back_filt_mode == 'norm':
                    self.back_mode = 'deconv'
                    self.back_filt_mode = 'raw'
                    self.back_stale = True
                else:
                    self.back_enabled = False
            elif tag == 'prev_ez_back_mode_loop':
                    # Cycle:
                    # off -> grad (raw) -> grad(gray) -> grad(norm) -> grad(normblur) -> deconv
                    if not self.back_enabled:
                        self.back_enabled = True
                        self.back_mode = 'deconv'
                        self.back_filt_mode = 'raw'
                        self.back_stale = True
                    elif self.back_mode == 'deconv':
                        self.back_mode = 'grad'
                        self.back_filt_mode = 'norm'
                        self.back_stale = True
                    elif self.back_mode == 'grad' and self.back_filt_mode == 'norm':
                        self.back_filt_mode = 'raw'
                    else:
                        self.back_enabled = False
            elif tag == 'freeze_back_unit':
                # Freeze selected layer/unit as backprop unit
                self.backprop_selection_frozen = not self.backprop_selection_frozen
                if self.backprop_selection_frozen:
                    # Grap layer/selected_unit upon transition from non-frozen -> frozen
                    self.backprop_layer_idx = self.layer_idx
                    self.backprop_unit = self.selected_unit                    
            elif tag == 'zoom_mode':
                self.layers_pane_zoom_mode = (self.layers_pane_zoom_mode + 1) % 3
                if self.layers_pane_zoom_mode == 2 and not self.back_enabled:
                    # Skip zoom into backprop pane when backprop is off
                    self.layers_pane_zoom_mode = 0

            elif tag == 'toggle_label_predictions':
                self.show_label_predictions = not self.show_label_predictions

            elif tag == 'toggle_unit_jpgs':
                self.show_unit_jpgs = not self.show_unit_jpgs

            elif tag == 'siamese_input_mode':
                self.siamese_input_mode = (self.siamese_input_mode + 1) % SiameseInputMode.NUMBER_OF_MODES

            elif tag == 'show_maximal_score':
                self.show_maximal_score = not self.show_maximal_score

            else:
                key_handled = False

            self._ensure_valid_selected()

            self.drawing_stale = key_handled   # Request redraw any time we handled the key

        return (None if key_handled else key)

    def redraw_needed(self):
        with self.lock:
            return self.drawing_stale

    def get_current_layer_definition(self):
        return self.settings.layers_list[self.layer_idx]

    def get_current_backprop_layer_definition(self):
        return self.settings.layers_list[self.backprop_layer_idx]

    def get_single_selected_data_blob(self, net, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_single_selected_data_blob(net, layer_def, self.siamese_input_mode)

    def get_single_selected_diff_blob(self, net, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_single_selected_diff_blob(net, layer_def, self.siamese_input_mode)

    def get_siamese_selected_data_blobs(self, net, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_siamese_selected_data_blobs(net, layer_def, self.siamese_input_mode)

    def get_siamese_selected_diff_blobs(self, net, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_siamese_selected_diff_blobs(net, layer_def, self.siamese_input_mode)


    def backward_from_layer(self, net, backprop_layer_def, backprop_unit):

        try:
            return SiameseHelper.backward_from_layer(net, backprop_layer_def, backprop_unit, self.siamese_input_mode)
        except AttributeError:
            print 'ERROR: required bindings (backward_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
            raise
        except ValueError:
            print "ERROR: probably impossible to backprop layer %s, ignoring to avoid crash" % (str(backprop_layer_def['name/s']))
            with self.lock:
                self.back_enabled = False

    def deconv_from_layer(self, net, backprop_layer_def, backprop_unit):

        try:
            return SiameseHelper.deconv_from_layer(net, backprop_layer_def, backprop_unit, self.siamese_input_mode)
        except AttributeError:
            print 'ERROR: required bindings (deconv_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
            raise
        except ValueError:
            print "ERROR: probably impossible to deconv layer %s, ignoring to avoid crash" % (str(backprop_layer_def['name/s']))
            with self.lock:
                self.back_enabled = False

    def get_default_layer_name(self, layer_def = None):

        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return self.siamese_helper.get_default_layer_name(layer_def)

    def siamese_input_mode_has_two_images(self, layer_def = None):
        '''
        helper function which checks whether the input mode is two images, and the provided layer contains two layer names
        :param layer: can be a single string layer name, or a pair of layer names
        :return: True if both the input mode is BOTH_IMAGES and layer contains two layer names, False oherwise
        '''
        # if no layer specified, get current layer
        if layer_def is None:
            layer_def = self.get_current_layer_definition()

        return SiameseHelper.siamese_input_mode_has_two_images(layer_def, self.siamese_input_mode)

    def move_selection(self, direction, dist = 1):

        default_layer_name = self.get_default_layer_name()
        default_top_name = layer_name_to_top_name(self.net, default_layer_name)

        if direction == 'left':
            if self.cursor_area == 'top':
                self.layer_idx = max(0, self.layer_idx - dist)
            else:
                self.selected_unit -= dist
        elif direction == 'right':
            if self.cursor_area == 'top':
                self.layer_idx = min(len(self.settings.layers_list) - 1, self.layer_idx + dist)
            else:
                self.selected_unit += dist
        elif direction == 'down':
            if self.cursor_area == 'top':
                self.cursor_area = 'bottom'
            else:
                self.selected_unit += self.net_blob_info[default_top_name]['tile_cols'] * dist
        elif direction == 'up':
            if self.cursor_area == 'top':
                pass
            else:
                self.selected_unit -= self.net_blob_info[default_top_name]['tile_cols'] * dist
                if self.selected_unit < 0:
                    self.selected_unit += self.net_blob_info[default_top_name]['tile_cols']
                    self.cursor_area = 'top'

    def _ensure_valid_selected(self):

        default_layer_name = self.get_default_layer_name()
        default_top_name = layer_name_to_top_name(self.net, default_layer_name)

        n_tiles = self.net_blob_info[default_top_name]['n_tiles']

        # Forward selection
        self.selected_unit = max(0, self.selected_unit)
        self.selected_unit = min(n_tiles-1, self.selected_unit)

        # Backward selection
        if not self.backprop_selection_frozen:
            # If backprop_selection is not frozen, backprop layer/unit follows selected unit
            if not (self.backprop_layer_idx == self.layer_idx and self.backprop_unit == self.selected_unit):
                self.backprop_layer_idx = self.layer_idx
                self.backprop_unit = self.selected_unit
                self.back_stale = True    # If there is any change, back diffs are now stale

    def fill_layers_list(self, net):

        # if layers list is empty, fill it with layer names
        if not self.settings.layers_list:

            # go over layers
            self.settings.layers_list = []
            for layer_name in list(net._layer_names):

                # skip inplace layers
                if len(net.top_names[layer_name]) == 1 and len(net.bottom_names[layer_name]) == 1 and net.top_names[layer_name][0] == net.bottom_names[layer_name][0]:
                    continue

                self.settings.layers_list.append( {'format': 'normal', 'name/s': layer_name} )

        # filter layers if needed
        if hasattr(self.settings, 'caffevis_filter_layers'):
            for layer_def in self.settings.layers_list:
                if self.settings.caffevis_filter_layers(layer_def['name/s']):
                    print '  Layer filtered out by caffevis_filter_layers: %s' % str(layer_def['name/s'])
            self.settings.layers_list = filter(lambda layer_def: not self.settings.caffevis_filter_layers(layer_def['name/s']), self.settings.layers_list)


        pass
