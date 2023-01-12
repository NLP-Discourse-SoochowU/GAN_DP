# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/3/13
@Description:
"""

import urllib
import gzip
import os
import shutil
import pickle as pkl


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def safe_mkdirs(path_list):
    """ Create a directory if there isn't one already. """
    for path in path_list:
        try:
            os.mkdir(path)
        except OSError:
            pass


def download_one_file(download_url,
                      local_dest,
                      expected_byte=None,
                      unzip_and_remove=False):
    """
    Download the file from download_url into local_dest
    if the file doesn't already exists.
    If expected_byte is provided, check if
    the downloaded file has the same number of bytes.
    If unzip_and_remove is True, unzip the file and remove the zip file
    """
    if os.path.exists(local_dest) or os.path.exists(local_dest[:-3]):
        print('%s already exists' % local_dest)
    else:
        print('Downloading %s' % download_url)
        local_file, _ = urllib.request.urlretrieve(download_url, local_dest)
        file_stat = os.stat(local_dest)
        if expected_byte:
            if file_stat.st_size == expected_byte:
                print('Successfully downloaded %s' % local_dest)
                if unzip_and_remove:
                    with gzip.open(local_dest, 'rb') as f_in, open(local_dest[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(local_dest)
            else:
                print('The downloaded file has unexpected number of bytes')


def save_data(obj, path, append=False):
    if append:
        with open(path, "wb+") as f:
            pkl.dump(obj, f)
    else:
        with open(path, "wb") as f:
            pkl.dump(obj, f)


def load_data(path):
    with open(path, "rb") as f:
        obj = pkl.load(f)
    return obj


def write_iterate(ite, file_path):
    with open(file_path, "w") as f:
        for line in ite:
            f.write(line + "\n")


def write_append(txt, file_path):
    with open(file_path, "a") as f:
        f.write(txt + "\n")


def write_over(txt, file_path):
    with open(file_path, "w") as f:
        f.write(txt + "\n")


def print_(str_, log_file, write_=False):
    if write_:
        write_over(str_, log_file)
    else:
        write_append(str_, log_file)
    print(str_)
