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


UC merced saved images are normalized to [0,1), so this needs to be fixed

References
----------
 (*) http://weegee.vision.ucmerced.edu/datasets/landuse.html

"""

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



def import_merced_data(self):

    data = pipe_utils.load_from_categorized_directory("/Volumes/insight/data/UCMerced_LandUse/Images")
    reduced_data = convert_classes(data)
    del data

    pipe_utils.generarate_train_and_test(
            reduced_data, 
            path="/Users/bdhammel/Documents/insight/harvesting/datasets/uc_merced/", 
            save=True
            )
