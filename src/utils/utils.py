import os
import shutil


def create_dir(dst_dir, overwrite=True):
    if os.path.exists(dst_dir):
        if overwrite:
            shutil.rmtree(dst_dir)
        else:
            raise IOError(dst_dir + ' already exists')
    os.mkdir(dst_dir)





