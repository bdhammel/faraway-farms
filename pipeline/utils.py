import pickle 
import numpy as np
from skimage import transform 
from skimage.external import tifffile
from skimage.io import imread
from skimage.util import view_as_blocks
from sklearn.model_selection import train_test_split
import glob
import os
import unittest
import math


# Conversion of labels to id for path classification
PATCH_CLASS_TO_ID = {
    'trees':0,
    'water':1,
    'crops':2,
    'vehicles':3,
    'buildings':4,
    'field':5
}

# Conversion of labels to id for object detection 
OBJ_CLASS_TO_ID = {
    'vehicles':0,
    'buildings':1,
    'animals':2,
    'trees':3
}


def ids_to_classes(ids):
    """Convert id integers back to a verbose label
    """

    ids = np.atleast_1d(ids)

    labels = []

    for _id in ids:
        for key, value in CLASS_TO_ID.items():
            if value == _id:
                labels.append(key)

    return labels


def dump_as_pickle(data, path):
    """Save a given python object as a pickle file

    save an object as a "obj.p"

    Args
    ----
    data (python object) : the object to save
    path (str) : the location to save the data, including the file name w/ 
        extension
    """

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickled_data(path):
    """Load in a pickled data file

    Args
    ----
    path (str) : path to the file to read

    Returns
    -------
    the data object
    """

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def read_raw_image(path, report=True):
    """Import a raw image 

    This could be as 8 bit or 16 bit... or 10 bit like some of the files...

    Args
    ----
    path (str) : path to the image file
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

    return img


def image_preprocessor(img):
    """Normalize the image

     - Convert 16 bit to 8 bit
     - Set color channel to the last channel


    Args
    ----
    img (np array) : raw image data
    channel_last (bool) : 

    Returns 
    -------
    numpy array of cleaned image data
    """

    data = np.asarray(img)

    # set the color channel to last if in channel_first format
    if len(data.shape) == 3 and data.shape[-1] != 3:
        data = np.rollaxis(data, 0, 3) 

    # if > 8 bit, shift to a 255 pixel max
    bitspersample = int(math.ceil(math.log(data.max(), 2)))
    if bitspersample > 8:
        data >>= bitspersample - 8
        data.astype('B')

    return data


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

    Warnings
    --------
    Not loading Train and Test sets from the files with contaminate your 
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
        Y += [CLASS_TO_ID[label]]*len(_x)
        X += _x
        del _x

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        np.reshape(X, (-1, 200, 200, 3)), 
        np.array(Y), 
        test_size=0.33, 
        random_state=42)


    if save and path is not None:
        dump_as_pickle(Xtrain, os.path.join(path, "xtrain.p"))
        dump_as_pickle(Xtest, os.path.join(path, "xtest.p"))
        dump_as_pickle(Ytrain, os.path.join(path, "ytrain.p"))
        dump_as_pickle(Ytest, os.path.join(path, "ytest.p"))


    return Xtrain, Xtest, Ytrain, Ytest


def chop_to_blocks(data, shape=()):
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
    assert shape[-1] == data.shape[-1]
    
    # Drop parts of the image that cant be captured by an integer number of 
    # strides 
    _split_factor = np.floor(np.divide(data.shape, shape)).astype(int)
    _img_lims = (_split_factor * shape)

    #print("Can only preserve up to pix: ", _img_lims)

    _data = np.ascontiguousarray(
                data[:_img_lims[0], :_img_lims[1], :_img_lims[2]]
            )

    return view_as_blocks(_data, shape) 


def as_batch(img, as_list=False):
    """

    Args
    ----
    """
    
    blocks = chop_to_blocks(img, shape=(200,200,3))
    og_shape = blocks.shape

    flat_blocks = blocks.reshape(np.prod(og_shape[:3]), *og_shape[3:])
    
    if as_list:
        return list(flat_blocks)
    else:
        return flat_blocks


def get_file_name_from_path(path):
    """Extract the file name from a given path

    if path is `/path/to/file/name.ext` then this functions returns `name`

    Args
    ----
    path (str) : path to a file of interest 

    Returns
    -------
    (str) The file name 
    """
    return os.path.splitext(os.path.basename(path))[0]

