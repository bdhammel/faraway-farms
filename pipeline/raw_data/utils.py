"""Base functions used to handle raw data being loaded, cleaned, and resaved
"""
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


def load_from_categorized_directory(path, load_labels):
    """Load in the raw image files into a dictionary

    The directory should contain folders for each specific class, with the 
    relevant images contained inside. 

    Args
    ----
    path (str) : the path to the directory where the images are stored

    Returns
    -------
    a dic of the images of form { 'label': [ [img], ...], ...}
    """
    data = {}

    for img_path in glob.glob(path + "/**/*"):

        *_, label, im_name = img_path.split(os.path.sep)

        if label in load_labels:
            img = read_image(img_path)
            img = clean_image(img)

        data.setdefault(label, []).append(img)

    return data

def generarate_train_and_test(data, path=None, save=False):
    """Take a reduced dataset and make train and test sets

    Warning!
    --------
    Not loading Train and Test sets from files will contaminate your 
    Test set with training data

    Args
    ----
    data (dict) : reduced data from convert_classes()
    save (bool) : weather or not to pickle the data

    Returns
    -------
    Xtrain, Xtest, Ytrain, Ytest
    """

    X = []
    Y = []

    for label in data.keys():
        _x = data[label]
        Y += [PATCH_CLASS_TO_ID[label]]*len(_x)
        X += _x
        del _x

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        np.reshape(X, (-1, 200, 200, 3)), 
        np.array(Y), 
        test_size=0.33, 
        random_state=42
    )


    if save and path is not None:
        dump_as_pickle(Xtrain, os.path.join(path, "xtrain.p"))
        dump_as_pickle(Xtest, os.path.join(path, "xtest.p"))
        dump_as_pickle(Ytrain, os.path.join(path, "ytrain.p"))
        dump_as_pickle(Ytest, os.path.join(path, "ytest.p"))


    return Xtrain, Xtest, Ytrain, Ytest
