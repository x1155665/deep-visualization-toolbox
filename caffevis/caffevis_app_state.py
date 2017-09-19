import os
import time
from threading import Lock


class SiameseInputMode:
    FIRST_IMAGE = 0
    SECOND_IMAGE = 1
    BOTH_IMAGES = 2
    NUMBER_OF_MODES = 3

class PatternMode:
    OFF = 0
    MAXIMAL_OPTIMIZED_IMAGE = 1
    MAXIMAL_INPUT_IMAGE = 2
    NUMBER_OF_MODES = 3

class CaffeVisAppState(object):
    '''State of CaffeVis app.'''

    def __init__(self, net, settings, bindings, net_layer_info):
        self.lock = Lock()  # State is accessed in multiple threads
        self.settings = settings
        self.bindings = bindings

        self._fill_headers_list(net, settings)

        if hasattr(self.settings, 'caffevis_filter_layers'):
            for name in self._headers:
                if self.settings.caffevis_filter_layers(name):
                    print '  Layer filtered out by caffevis_filter_layers: %s' % name
            self._headers = filter(lambda name: not self.settings.caffevis_filter_layers(name), self._headers)
        self.net_layer_info = net_layer_info
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

    @staticmethod
    def get_header_from_layer(layer):
        '''
        helper function which returns the header name, given a single layer, or layer pair
        :param layer: can be either single layer string, or apir of layers
        :return: header for layer
        '''

        # if we have only a single layer, the header is the layer name
        if type(layer) is str:
            return layer

        # if we got a pair of layers
        elif (type(layer) is tuple) and (len(layer) == 2):
            # build header in format: common_prefix + first_postfix | second_postfix
            prefix = os.path.commonprefix(layer)
            prefix_len = len(prefix)
            postfix0 = layer[0][prefix_len:]
            postfix1 = layer[1][prefix_len:]
            header_name = '%s%s|%s' % (prefix, postfix0, postfix1)
            return header_name

    def _fill_headers_list(self, net, settings):
        '''
        helper function which fills the headers list, using either the entire blob names, or the siamese layers definition
        :param net: network to generate headers for
        :param settings: settings holder
        :return: N/A
        '''

        # if loading a siamese network, and layers list is not empty
        if settings.is_siamese and settings.siamese_layers_list:

            self._headers = list()
            for item in settings.siamese_layers_list:
                self._headers.append(self.get_header_from_layer(item))

        else:
            self._headers = net.blobs.keys()
            self._headers = self._headers[1:]  # chop off data layer

    def _reset_user_state(self):
        self.siamese_input_mode = SiameseInputMode.BOTH_IMAGES
        self.show_maximal_score = False
        self.layer_idx = 0
        self._update_layer_name()
        self.layer_boost_indiv_idx = self.settings.caffevis_boost_indiv_default_idx
        self.layer_boost_indiv = self.layer_boost_indiv_choices[self.layer_boost_indiv_idx]
        self.layer_boost_gamma_idx = self.settings.caffevis_boost_gamma_default_idx
        self.layer_boost_gamma = self.layer_boost_gamma_choices[self.layer_boost_gamma_idx]
        self.cursor_area = 'top'   # 'top' or 'bottom'
        self.selected_unit = 0
        # Which layer and unit (or channel) to use for backprop
        self.backprop_layer = self.layer
        self.backprop_unit = self.selected_unit
        self.backprop_selection_frozen = False    # If false, backprop unit tracks selected unit
        self.back_enabled = False
        self.back_mode = 'grad'      # 'grad' or 'deconv'
        self.back_filt_mode = 'raw'  # 'raw', 'gray', 'norm', 'normblur'
        self.pattern_mode = PatternMode.OFF    # type of patterns to show instead of activations in layers pane: maximal optimized image, maximal input image, off
        self.pattern_first_only = True         # should we load only the first pattern image for each neuron, or all the relevant images per neuron
        self.layers_pane_zoom_mode = 0       # 0: off, 1: zoom selected (and show pref in small pane), 2: zoom backprop
        self.layers_show_back = False   # False: show forward activations. True: show backward diffs
        self.show_label_predictions = self.settings.caffevis_init_show_label_predictions
        self.show_unit_jpgs = self.settings.caffevis_init_show_unit_jpgs
        self.drawing_stale = True
        kh,_ = self.bindings.get_key_help('help_mode')
        self.extra_msg = '%s for help' % kh[0]

    def _update_layer_name(self):
        '''
        updates the selected layer, after layer_idx got updated
        in Siamese networks, returns the relevant layer according to current input mode
        on non-siamese networks, header name is the layer name
        :return: N/A
        '''

        if self.settings.is_siamese and (self.settings.siamese_layers_list) and \
                (type(self.settings.siamese_layers_list[self.layer_idx]) is tuple) and \
                (len(self.settings.siamese_layers_list[self.layer_idx]) == 2):

            if self.siamese_input_mode == SiameseInputMode.FIRST_IMAGE:
                self.layer = self.settings.siamese_layers_list[self.layer_idx][0]
            elif self.siamese_input_mode == SiameseInputMode.SECOND_IMAGE:
                self.layer = self.settings.siamese_layers_list[self.layer_idx][1]
            elif self.siamese_input_mode == SiameseInputMode.BOTH_IMAGES:
                self.layer = self.settings.siamese_layers_list[self.layer_idx]
        else:
            self.layer = self._headers[self.layer_idx]

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
                self._update_layer_name()

            elif tag == 'sel_layer_right':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = min(len(self._headers) - 1, self.layer_idx + 1)
                self._update_layer_name()

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
                if self.pattern_mode != PatternMode.OFF and not hasattr(self.settings, 'caffevis_unit_jpg_dir'):
                    print 'Cannot switch to pattern mode; caffevis_unit_jpg_dir not defined in settings.py.'
                    self.pattern_mode = PatternMode.OFF
            elif tag == 'prev_pattern_mode':
                self.pattern_mode = (self.pattern_mode - 1 + PatternMode.NUMBER_OF_MODES) % PatternMode.NUMBER_OF_MODES
                if self.pattern_mode != PatternMode.OFF and not hasattr(self.settings, 'caffevis_unit_jpg_dir'):
                    print 'Cannot switch to pattern mode; caffevis_unit_jpg_dir not defined in settings.py.'
                    self.pattern_mode = PatternMode.OFF
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
                    self.backprop_layer = self.layer
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

    def get_default_layer_name(self):
        '''
        helper function which returns the first layer in the layer tuple, or the only one
        many parts of the code can take info from first layer, since it should be the same as second layer info
        :return: layer name which can be used to get information which is not input specific
        '''
        return self.get_default_layer_name_for_layer(self.layer)

        return default_layer_name


    def get_default_layer_name_for_layer(self, layer):
        '''
        helper function which returns the first layer in the layer tuple, or the only one
        many parts of the code can take info from first layer, since it should be the same as second layer info
        :return: layer name which can be used to get information which is not input specific
        '''

        if self.siamese_input_mode_has_two_images(layer):
            default_layer_name = layer[0]
        elif self.siamese_input_mode == SiameseInputMode.FIRST_IMAGE and self.is_pair_of_layers(layer):
            default_layer_name = layer[0]
        elif self.siamese_input_mode == SiameseInputMode.SECOND_IMAGE and self.is_pair_of_layers(layer):
            default_layer_name = layer[1]
        else:
            default_layer_name = layer

        return default_layer_name

    @staticmethod
    def is_pair_of_layers(layer):
        return (type(layer), len(layer)) == (tuple, 2)

    def siamese_input_mode_has_two_images(self, layer):
        '''
        helper function which checks whether the input mode is two images, and the provided layer contains two layer names
        :param layer: can be a single string layer name, or a pair of layer names
        :return: True if both the input mode is BOTH_IMAGES and layer contains two layer names, False oherwise
        '''

        return self.siamese_input_mode == SiameseInputMode.BOTH_IMAGES and self.is_pair_of_layers(layer)

    def move_selection(self, direction, dist = 1):

        default_layer_name = self.get_default_layer_name()

        if direction == 'left':
            if self.cursor_area == 'top':
                self.layer_idx = max(0, self.layer_idx - dist)
                self._update_layer_name()
            else:
                self.selected_unit -= dist
        elif direction == 'right':
            if self.cursor_area == 'top':
                self.layer_idx = min(len(self._headers) - 1, self.layer_idx + dist)
                self._update_layer_name()
            else:
                self.selected_unit += dist
        elif direction == 'down':
            if self.cursor_area == 'top':
                self.cursor_area = 'bottom'
            else:
                self.selected_unit += self.net_layer_info[default_layer_name]['tile_cols'] * dist
        elif direction == 'up':
            if self.cursor_area == 'top':
                pass
            else:
                self.selected_unit -= self.net_layer_info[default_layer_name]['tile_cols'] * dist
                if self.selected_unit < 0:
                    self.selected_unit += self.net_layer_info[default_layer_name]['tile_cols']
                    self.cursor_area = 'top'

    def _ensure_valid_selected(self):

        default_layer_name = self.get_default_layer_name()

        n_tiles = self.net_layer_info[default_layer_name]['n_tiles']

        # Forward selection
        self.selected_unit = max(0, self.selected_unit)
        self.selected_unit = min(n_tiles-1, self.selected_unit)

        # Backward selection
        if not self.backprop_selection_frozen:
            # If backprop_selection is not frozen, backprop layer/unit follows selected unit
            if not (self.backprop_layer == self.layer and self.backprop_unit == self.selected_unit):
                self.backprop_layer = self.layer
                self.backprop_unit = self.selected_unit
                self.back_stale = True    # If there is any change, back diffs are now stale
