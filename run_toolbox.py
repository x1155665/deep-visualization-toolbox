#! /usr/bin/env python

# this import must comes first to make sure we use the non-display backend
import matplotlib
matplotlib.use('Agg')

import os
from live_vis import LiveVis
from bindings import bindings
try:
    import settings
except:
    print '\nError importing settings.py. Check the error message below for more information.'
    print "If you haven't already, you'll want to open the settings_model_selector.py file"
    print 'and edit it to point to your caffe checkout.\n'
    raise


def main():
    lv = LiveVis(settings)

    help_keys, _ = bindings.get_key_help('help_mode')
    quit_keys, _ = bindings.get_key_help('quit')
    print '\n\nRunning toolbox. Push %s for help or %s to quit.\n\n' % (help_keys[0], quit_keys[0])
    lv.run_loop()


    
if __name__ == '__main__':
    main()
