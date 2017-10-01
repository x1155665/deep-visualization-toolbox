import time
import cv2
import numpy as np

from codependent_thread import CodependentThread
from misc import WithTimer
from caffevis_helper import net_preproc_forward
from image_misc import resize_without_fit
from caffevis_app_state import BackpropMode

class CaffeProcThread(CodependentThread):
    '''Runs Caffe in separate thread.'''

    def __init__(self, settings, net, state, loop_sleep, pause_after_keys, heartbeat_required, mode_gpu):
        CodependentThread.__init__(self, heartbeat_required)
        self.daemon = True
        self.net = net
        self.input_dims = self.net.blobs['data'].data.shape[2:4]    # e.g. (227,227)
        self.state = state
        self.last_process_finished_at = None
        self.last_process_elapsed = None
        self.frames_processed_fwd = 0
        self.frames_processed_back = 0
        self.loop_sleep = loop_sleep
        self.pause_after_keys = pause_after_keys
        self.debug_level = 0
        self.mode_gpu = mode_gpu      # Needed so the mode can be set again in the spawned thread, because there is a separate Caffe object per thread.
        self.settings = settings


    def run(self):
        print 'CaffeProcThread.run called'
        frame = None

        import caffe
        # Set the mode to CPU or GPU. Note: in the latest Caffe
        # versions, there is one Caffe object *per thread*, so the
        # mode must be set per thread! Here we set the mode for the
        # CaffeProcThread thread; it is also set in the main thread.
        if self.mode_gpu:
            caffe.set_mode_gpu()
            print 'CaffeVisApp mode (in CaffeProcThread): GPU'
        else:
            caffe.set_mode_cpu()
            print 'CaffeVisApp mode (in CaffeProcThread): CPU'
        
        while not self.is_timed_out():
            with self.state.lock:
                if self.state.quit:
                    #print 'CaffeProcThread.run: quit is True'
                    #print self.state.quit
                    break
                    
                #print 'CaffeProcThread.run: caffe_net_state is:', self.state.caffe_net_state

                #print 'CaffeProcThread.run loop: next_frame: %s, caffe_net_state: %s, back_enabled: %s' % (
                #    'None' if self.state.next_frame is None else 'Avail',
                #    self.state.caffe_net_state,
                #    self.state.back_enabled)

                frame = None
                run_fwd = False
                run_back = False
                if self.state.caffe_net_state == 'free' and time.time() - self.state.last_key_at > self.pause_after_keys:
                    frame = self.state.next_frame
                    self.state.next_frame = None
                    back_enabled = self.state.back_enabled
                    back_mode = self.state.back_mode
                    back_stale = self.state.back_stale
                    backprop_layer_def = self.state.get_current_backprop_layer_definition()
                    backprop_unit = self.state.backprop_unit

                    # Forward should be run for every new frame
                    run_fwd = (frame is not None)
                    # Backward should be run if back_enabled and (there was a new frame OR back is stale (new backprop layer/unit selected))
                    run_back = (back_enabled and (run_fwd or back_stale))
                    self.state.caffe_net_state = 'proc' if (run_fwd or run_back) else 'free'

            #print 'run_fwd,run_back =', run_fwd, run_back
            
            if run_fwd:
                #print 'TIMING:, processing frame'
                self.frames_processed_fwd += 1

                if self.settings.is_siamese and ((type(frame), len(frame)) == (tuple, 2)):
                    im_small = self.state.convert_image_pair_to_network_input_format(self.settings, frame, self.input_dims)

                else:
                    im_small = resize_without_fit(frame, self.input_dims)

                with WithTimer('CaffeProcThread:forward', quiet = self.debug_level < 1):
                    net_preproc_forward(self.settings, self.net, im_small, self.input_dims)

            if run_back:

                if back_mode == BackpropMode.GRAD:
                    with WithTimer('CaffeProcThread:backward', quiet = self.debug_level < 1):
                        self.state.backward_from_layer(self.net, backprop_layer_def, backprop_unit)

                elif back_mode == BackpropMode.DECONV_ZF:
                    with WithTimer('CaffeProcThread:deconv', quiet = self.debug_level < 1):
                        self.state.deconv_from_layer(self.net, backprop_layer_def, backprop_unit, 'Zeiler & Fergus')

                elif back_mode == BackpropMode.DECONV_GB:
                    with WithTimer('CaffeProcThread:deconv', quiet=self.debug_level < 1):
                        self.state.deconv_from_layer(self.net, backprop_layer_def, backprop_unit, 'Guided Backprop')

                with self.state.lock:
                    self.state.back_stale = False

            if run_fwd or run_back:
                with self.state.lock:
                    self.state.caffe_net_state = 'free'
                    self.state.drawing_stale = True
                now = time.time()
                if self.last_process_finished_at:
                    self.last_process_elapsed = now - self.last_process_finished_at
                self.last_process_finished_at = now
            else:
                time.sleep(self.loop_sleep)
        
        print 'CaffeProcThread.run: finished'
        print 'CaffeProcThread.run: processed %d frames fwd, %d frames back' % (self.frames_processed_fwd, self.frames_processed_back)

    def approx_fps(self):
        '''Get the approximate frames per second processed by this
        thread, considering only the last image processed. If more
        than two seconds ago, assume pipeline has stalled elsewhere
        (perhaps using static images that are only processed once).
        '''
        if self.last_process_elapsed and (time.time() - self.last_process_finished_at) < 2.0:
            return 1.0 / (self.last_process_elapsed + 1e-6)
        else:
            return 0.0
