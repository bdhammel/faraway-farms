import pickle 
import numpy as np

CLASS_TO_ID = {
    'trees':0,
    'water':1,
    'crops':2,
    'vehicles':3,
    'buildings':4
}


def ids_to_classes(ids):

    ids = np.atleast_1d(ids)

    labels = []

    for _id in ids:
        for key, value in CLASS_TO_ID.items():
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
