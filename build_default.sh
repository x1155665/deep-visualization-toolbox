#!/usr/bin/env bash
cd caffe
# if Makefile.config has already existed, then don't overwrite it
if [ ! -e  Makefile.config ]; then
    cp Makefile.config.example Makefile.config
fi
make all pycaffe
cd ..
cp settings_user.py.example settings_user.py
