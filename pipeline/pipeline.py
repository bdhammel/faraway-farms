import numpy as np

from skimage.io import imread
from skimage.external import tifffile
from skimage.util import view_as_blocks

import os


class SatelliteImage:


    def __init__(self, data):
        """

        Args
        ----
        image_id (str) : 
        data (nparray) : Satellite image of the form H x W x Ch
        features (dict)
        """
        self._data = data/data.max()


    def _preprocess(self, data):
        pass


    @property
    def data(self):
        return self._data


    @property
    def block_shape(self):
        return self._og_shape


    def _chop_to_blocks(self, shape=()):
        """Subdivides the current image and returns an array of DataFrame images 
        with the dims `shape`

        Args
        ----
        shape (tuple : ints) : the dims of the subdivided images

        Returns
        -------
        (list : DataFrame)
        """

        # Make sure there are not multiple strides in the color ch direction 
        assert shape[-1] == self.data.shape[-1]
        
        # Drop parts of the image that cant be captured by an integer number of 
        # strides 
        _split_factor = np.floor(np.divide(self.data.shape, shape)).astype(int)
        _img_lims = (_split_factor * shape)

        print("Can only preserve up to pix: ", _img_lims)

        _data = np.ascontiguousarray(
                self._data[:_img_lims[0], :_img_lims[1], :_img_lims[2]]
                )

        return view_as_blocks(_data, shape) 


    def as_batch(self):
        
        blocks = self._chop_to_blocks(shape=(200,200,3))
        self._og_shape = blocks.shape

        return blocks.reshape(np.prod(self._og_shape[:3]), *self._og_shape[3:])



def re_stitch(img_batch, og_shape):
    pass


def load_from_file(path):
    """
    """

    ext = os.path.splitext(path)[1]

    if ext in [".jpg", ".png"]:
        _img = imread(path)
    elif ext == ".tif":
        _img = tifffile.imread(path)
    else:
        raise Exception("Not a supported extension")

    img = SatelliteImage(_img)
    del _img

    return img

    
def load_from_link(path):
    pass

