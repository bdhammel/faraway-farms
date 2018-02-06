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

# Convert the WHU RS19 labels to labels used in this project
# This is somewhat of an arbitrary mapping. Care should be taken to ensure 
# that labels are constant across files (e.g. clean_uc_merced, clean_whu_rs19)
# if outputs are combined
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

