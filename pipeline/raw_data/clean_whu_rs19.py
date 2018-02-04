"""
WHU_RS19 Dataset

600x600x3 images .jpg format
unit8 dtype i.e. values [0,255]


    Airport
    Beach
    Bridge
    Commercial
    Desert
    Farmland
    Forest
    Industrial
    Meadow
    Mountain
    Park
    Parking
    Pond
    Port
    Residential
    River
    Viaduct
    footballField
    railwayStation
"""

import numpy as np
from pipeline import utils
import glob
import os


# Convert the WHU RS19 labels to labels used in this project
MAP_TO_LOCAL_LABELS = {
        'Airport':None,
        'Beach':None,
        'Bridge':None,
        'Commercial':None,
        'Desert':None,
        'Farmland':'crops',
        'Forest':'trees',
        'Industrial':None,
        'Meadow':'field',
        'Mountain':None,
        'Park':None,
        'Parking':'vehicles',
        'Pond':'water',
        'Port':None,
        'Residential':'buildings',
        'River':None,
        'Viaduct':None,
        'footballField':None,
        'railwayStation':None
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

    for label, images in raw_data.items():
        local_label = MAP_TO_LOCAL_LABELS[label]
        if label:
            data[local_label] = images

    return data


def clean_image(img):

    # Normalize the image
    assert img.max() < 256
    img = img/255

    # split image to 200, 200, 3
    imgs = utils.as_batch(img, as_list=True)

    return imgs


def import_whu_rs19(path):

    labels_to_import = [label for label, local_label in MAP_TO_LOCAL_LABELS.items() if local_label]
    
    data = {}

    for img_path in glob.glob(path + "/**/*"):

        *_, label, im_name = img_path.split(os.path.sep)

        if label in labels_to_import:
            try:
                img = utils.read_image(img_path)
            except Exception:
                pass
            else:
                img = clean_image(img)
                data[label] = data.setdefault(label, []) + img

    data = convert_classes(data)

    return data


    
