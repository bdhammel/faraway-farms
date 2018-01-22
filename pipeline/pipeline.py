import numpy as np
from pipeline import utils

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


    def as_batch(self):
        return utils.as_batch(self._data)


def re_stitch(img_batch, og_shape):
    pass


def load_from_file(path):
    """
    """
    img = SatelliteImage(utils.read_image(path))

    return img

    
def load_from_link(path):
    pass


