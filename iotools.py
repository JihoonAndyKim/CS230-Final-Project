# Input/output functions borrowed from Jun-Yan Zhu's
# CycleGAN github.

import copy
import os
import shutil

import numpy as np

import torch

# Directory creation (multiple directories)
def create_dirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            create_dir(path)
    else:
        create_dir(paths)

# Directory creation
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Save checkpoint to restore execution if halted
def checkpoint_save(state, save_path):

    # Save model state
    torch.save(state, save_path)

    save_dir  = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'checkpoints_list')
    save_path = os.path.basename(save_path)

    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    for ckpt in ckpt_list[2:]:
        ckpt = os.path.join(save_dir, ckpt[:-1])
        if os.path.exists(ckpt):
            os.remove(ckpt)
        ckpt_list[2:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

# Load checkpoint to restore execution if halted
def checkpoint_load(ckpt_dir_or_file, map_location=None):
    if os.path.isdir(ckpt_dir_or_file):
        with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
            ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print('Successfully loaded checkpoint from %s' % ckpt_path)
    return ckpt

# Create filesystem structure for data
def create_fs_structure(data_dir):
    dirs = {}
    dirs['trainA'] = os.path.join(data_dir, 'link_trainA')
    dirs['trainB'] = os.path.join(data_dir, 'link_trainB')
    dirs['testA'] = os.path.join(data_dir, 'link_testA')
    dirs['testB'] = os.path.join(data_dir, 'link_testB')
    create_dirs(dirs['trainA'])
    create_dirs(dirs['trainB'])
    create_dirs(dirs['testA'])
    create_dirs(dirs['testB'])

    for key in dirs:
        try:
            os.remove(os.path.join(dirs[key], '0'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(data_dir, key)),
                   os.path.join(dirs[key], '0'))

    return dirs
