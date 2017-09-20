
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

caffevis_caffe_root = base_folder + '/caffe'

if model_to_load == 'caffenet_yos':
    from models.caffenet_yos.settings_caffenet_yos import *

elif model_to_load == 'other_model':
    from models.other_model.settings_other_model import *

