import numpy as np
from skimage.external import tifffile

import os, sys

PROJ_DIR = "/Users/bdhammel/Documents/insight/harvesting/"

if PROJ_DIR not in sys.path:
    sys.path.append(PROJ_DIR)

from pipeline import utils


def load(path):
    """Load in the raw image files into a dictionary

    The directory should contain folders for each specific class, with the 
    relevant images contained inside. Images need to be .tif

    Args
    ----
    path (str) : the path to the directory where the images are stored

    Returns
    -------
    a dic of the images of form { 'label': [ [img], ...], ...}
    """
    data = {}

    for img_path in glob.glob(path):
        *_, label, im_name = img_path.split(os.path.sep)
        
        img = tifffile.imread(img_path)

        if img.max() < 1.0:
            img /= img.max()

        img = transform.resize(img, (200, 200, 3), mode='reflect')
        data.setdefault(label, []).append(img)

    return data


def convert_classes(raw_data):
    """Convert the MC Land use classes to the specific things I'm interested in 

    Args
    ----
    raw_data (dict) : dictionary of raw data, gotten from load_raw_data()

    Returns
    -------
    Similar dictionary but with labels of specific interest 
    """

    data = {
        'trees':raw_data['forest'],
        'water':raw_data['river'],
        'crops':raw_data['agricultural'],
        'vehicles':raw_data['parkinglot'],
        'buildings':raw_data['sparseresidential']
    }

    return data


def generarate_train_and_test(data, save=False):
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

    if save:
        utils.dump_as_pickle(Xtrain, "./xtrain.p")
        utils.dump_as_pickle(Xtest, "./xtest.p")
        utils.dump_as_pickle(Ytrain, "./ytrain.p")
        utils.dump_as_pickle(Ytest, "./ytest.p")

    return Xtrain, Xtest, Ytrain, Ytest


if __name__ == '__main__':

    raise Exception("This needs to be tested")





