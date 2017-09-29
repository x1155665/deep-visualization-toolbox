
# Import user settings. Turn off creation of X.pyc to avoid stale settings if X.py is removed.
import os
import sys
sys.dont_write_bytecode = True
try:
    from settings_user import *
except ImportError:
    if not os.path.exists('settings_user.py'):
        raise Exception('Could not import settings_user. Did you create it from the template? Start with:\n\n  $ cp settings_user_template.py settings_user.py')
    else:
        raise
# Resume usual pyc creation
sys.dont_write_bytecode = False

caffevis_caffe_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'./caffe')

# the following code runs dynamically the import command:
# from models.YOUR_MODEL.settings_YOUR_MODEL import *
import_code = 'from models.' + model_to_load + '.settings_' + model_to_load + ' import *'
exec (import_code, globals())
