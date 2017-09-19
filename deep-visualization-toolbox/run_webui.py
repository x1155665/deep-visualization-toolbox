#! /usr/bin/env python

import os
import thread
from live_vis import LiveVis
from bindings import bindings
try:
    import settings
except:
    print '\nError importing settings.py. Check the error message below for more information.'
    print "If you haven't already, you'll want to copy one of the settings_local.template-*.py files"
    print 'to settings_local.py and edit it to point to your caffe checkout. E.g. via:'
    print
    print '  $ cp models/caffenet_yos/settings_local_template_caffenet_yos.py settings_local.py'
    print '  $ < edit settings_local.py >\n'
    raise

if not os.path.exists(settings.caffevis_caffe_root):
    raise Exception('ERROR: Set caffevis_caffe_root in settings.py first.')

import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while True:
        frame = get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def get_frame():
    # We are using Motion JPEG, but OpenCV defaults to capture raw images,
    # so we must encode it into JPEG in order to correctly display the
    # video stream.

    global lv

    ret, jpeg = cv2.imencode('.jpg', lv.window_buffer[:,:,::-1])
    return jpeg.tobytes()


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    global lv

    def someFunc():
        print "someFunc was called"
        lv.run_loop()


    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        # The reloader has already run - do what you want to do here

        lv = LiveVis(settings)
        help_keys, _ = bindings.get_key_help('help_mode')
        quit_keys, _ = bindings.get_key_help('quit')
        print '\n\nRunning toolbox. Push %s for help or %s to quit.\n\n' % (help_keys[0], quit_keys[0])

        thread.start_new_thread(someFunc, ())

    app.run(host='127.0.0.1', debug=True)
