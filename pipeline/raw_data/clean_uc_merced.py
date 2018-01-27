"""Importing and cleaning from the UC Merced dataset

UC Merced Classes:
    agricultural
    airplane
    baseballdiamond
    beach
    buildings
    chaparral
    denseresidential
    forest
    freeway
    golfcourse
    harbor
    intersection
    mediumresidential
    mobilehomepark
    overpass
    parkinglot
    river
    runway
    sparseresidential
    storagetanks
    tenniscourt

UC merced images seem to have already been normalized to [0,1)

"""

import numpy as np
from pipeline import utils


MAP_TO_LOCAL_LABELS = {
        'agricultural':'crops',
        'airplane':None,
        'baseballdiamond':None,
        'beach':None,
        'buildings':None,
        'chaparral':'field',
        'denseresidential':None,
        'forest':'trees',
        'freeway':None,
        'golfcourse':None,
        'harbor':None,
        'intersection':None,
        'mediumresidential':None,
        'mobilehomepark':None,
        'overpass':None,
        'parkinglot':'vehicles',
        'river':'water',
        'runway':None,
        'sparseresidential':'buildings',
        'storagetanks':None,
        'tenniscourt':None
}


def convert_classes(raw_data):
    """Convert the MC Land use classes to the specific things I'm interested in 

    Args
    ----
    raw_data (dict) : dictionary of raw data, gotten from load_raw_data()

    Returns
    -------
    Similar dictionary but with labels of specific interest 
    """

    data = {}

    for merced_label, images in raw_data.items():
        label = MAP_TO_LOCAL_LABELS[merced_label]
        if label:
            data[label] = images

    return data


def import_merced_data(self):
    from importlib import reload
    reload(utils)

    data = utils.load_from_categorized_directory("/Volumes/insight/data/UCMerced_LandUse/Images")
    reduced_data = convert_classes(data)
    del data

    utils.generarate_train_and_test(
            reduced_data, 
            path="/Users/bdhammel/Documents/insight/harvesting/datasets/uc_merced/", 
            save=True
            )


if __name__ == '__main__':
    #import_merced_data()
    pass





