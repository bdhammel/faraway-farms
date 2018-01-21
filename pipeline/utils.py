import pickle 
import numpy as np
from skimage import transform 
from skimage.external import tifffile
from skimage.io import imread
from sklearn.model_selection import train_test_split
import glob
import os
import unittest

CLASS_TO_ID = {
    'trees':0,
    'water':1,
    'crops':2,
    'vehicles':3,
    'buildings':4,
    'field':5
}


def ids_to_classes(ids):

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


def read_image(path):
    ext = os.path.splitext(path)[1]

    if ext in [".jpg", ".png"]:
        img = imread(path)
    elif ext == ".tif":
        img = tifffile.imread(path)
    else:
        raise Exception("{} Not a supported extension".format(ext))

    return img


def clean_image(img, size=(200,200,3)):

    img = transform.resize(img, size, mode='reflect')

    if img.max() > 1 or img.min() < 0:
        raise Exception("image needs to be cleaned")

    return img


def load_from_categorized_directory(path):
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
        np.array(X), np.array(Y), test_size=0.33, random_state=42)

    if save and path is not None:
        dump_as_pickle(Xtrain, os.path.join(path, "xtrain.p"))
        dump_as_pickle(Xtest, os.path.join(path, "xtest.p"))
        dump_as_pickle(Ytrain, os.path.join(path, "ytrain.p"))
        dump_as_pickle(Ytest, os.path.join(path, "ytest.p"))

    return Xtrain, Xtest, Ytrain, Ytest





