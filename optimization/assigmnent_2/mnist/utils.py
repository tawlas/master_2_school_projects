import six
from six.moves.urllib.request import urlretrieve, urlopen
from six.moves.urllib.error import HTTPError, URLError
import os
from pathlib import Path
import numpy as np
    
def load_mnist():
    path = get_file('mnist.npz', 'https://s3.amazonaws.com/img-datasets/mnist.npz','8a61469f7ea1b51cbae51d4f78837e45')
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)


def progress_bar(count, block_size, total_size):
    total = total_size // block_size
    if (count % (total//20)) == 0:
        i = round(count / total * 100)
        print("[%-20s]" % ('='*(i//5)), "%d%%" % i, end="\r")

        
def get_file(fname, origin, file_hash=None):
    root = Path().absolute() # os.path.expanduser('~')
    data_dir = os.path.join(root, 'mnist')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fpath = os.path.join(data_dir, fname)
    download = False if os.path.exists(fpath) else True
    if download:
        print('Downloading data from', origin)
        try:
            try:
                urlretrieve(origin, fpath, progress_bar)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
    return fpath
    
