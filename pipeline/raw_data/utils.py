import numpy as np
from PIL import Image, ImageDraw
from skimage.external import tifffile
from skimage.io import imread
import os

from pipeline import utils as pipe_utils


class RawPatchImage:
    """This class is currently not used
    """
    pass


class RawObjImage(pipe_utils.SatelliteImage):
    """
    """

    def __init__(self, image_path):
        self._image_id = pipe_utils.get_file_name_from_path(image_path)

        # import the data for the image and the json
        print("loading image ", self._image_id)
        _data = read_raw_image(image_path)
        self._data = pipe_utils.image_preprocessor(_data)
        del _data
        print("...done")
        self._features = {}


    @property
    def features(self):
        return self._features


    def append_feature(self, label, coor):
        """Connect a feature to the imported image

        Args
        ----
        label (str) : label of the feature 
        """
        self._features.setdefault(label, []).append(coor)


    def has_labels(self):
        """Return the labels present in this image

        Returns
        -------
        (list : str) : a list of the label names
        """
        return list(self.features.keys())


    def show(self, labels=[], as_poly=True):
        """Display the satellite image with the feature overlay

        Args
        ----
        labels (list) : list of ints corresponding to feature names
        """

        labels = pipe_utils.atleast_list(labels)

        im = Image.fromarray(self.data)
        draw = ImageDraw.Draw(im)       

        for label in labels:
            locs = self.get_features(as_poly=as_poly)[label]

            for coors in locs:
                if as_poly:
                    draw.polygon(
                            coors,
                            outline='red'
                    )
                else:
                    draw.rectangle(
                            coors,
                            outline='red'
                    )

        im.show()


def read_raw_image(path, report=True, check_data=False):
    """Import a raw image 

    This could be as 8 bit or 16 bit... or 10 bit like some of the files...

    Args
    ----
    path (str) : path to the image file
    report (bool) : output a short log on the imported data 

    Raises
    ------
    Exception : If the image passed does not have a file extension that's expected

    Returns 
    -------
    numpy array of raw image 
    """
    ext = os.path.splitext(path)[1]

    if ext in [".jpg", ".png"]:
        img = imread(path)
    elif ext == ".tif":
        img = tifffile.imread(path)
    else:
        raise Exception("{} Not a supported extension".format(ext))

    if report:
        print("Image {} loaded".format(os.path.basename(path)))
        print("\tShape: ", img.shape)
        print("\tdtype: ", img.dtype)
        print("Values: ({:.2f},{:.2f})".format(img.min(), img.max())), 

    if check_data:  
        data_is_ok(img, raise_exception=True)
       
    return img


