import os
import shutil
deconv_dir = 'unit_jpg_vis/max_deconv'
maxim_dir = 'unit_jpg_vis/max_im'
opt_dir = 'unit_jpg_vis/regularized_opt'
output_dir = 'outputs'
layer_info = {'conv1':96, 'conv2':256, 'conv3':384, 'conv4':384, 'conv5':256, 'fc8':1000, 'prob':1000}
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for key, value in layer_info.iteritems():
    deconv_layer_dir = os.path.join(deconv_dir, key)
    maxim_layer_dir = os.path.join(maxim_dir, key)
    opt_layer_dir = os.path.join(opt_dir, key)
    dst_layer_dir = os.path.join(output_dir, key)
    if not os.path.exists(dst_layer_dir):
        os.mkdir(dst_layer_dir)
    for unit_id in range(value):
        unit_dir = os.path.join(dst_layer_dir, "unit_%04d" % unit_id)
        if not os.path.exists(unit_dir):
            os.mkdir(unit_dir)
        deconv_unit_name = os.path.join(deconv_layer_dir, '%s_%04d.jpg' % (key, unit_id))
        opt_unit_name = os.path.join(opt_layer_dir, '%s_%04d_montage.jpg' % (key, unit_id))
        maxim_unit_name = os.path.join(maxim_layer_dir, '%s_%04d.jpg' % (key, unit_id))
        shutil.copyfile(deconv_unit_name, os.path.join(unit_dir, 'deconv.png'))
        shutil.copyfile(opt_unit_name, os.path.join(unit_dir, 'opt.jpg'))
        shutil.copyfile(maxim_unit_name, os.path.join(unit_dir, 'maxim.png'))
    print("%s finished" % key)
