"""This script handles pulling in the data from the kaggle competition from DSTL, cleaning it, and saving it in a format that's appropriate to be read in by the keras-retinanet model


References
----------
(*) https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection

"""
import matplotlib.pyplot as plt
from matplotlib import patches
import skimage.external.tifffile as tif
import numpy as np 
import glob
import geojson
import pandas as pd
import os
import re

from pipeline import utils as pipe_utils


MAP_TO_LOCAL_LABELS = {
    '001_MM_L2_LARGE_BUILDING':'buildings',
    '001_MM_L3_RESIDENTIAL_BUILDING':'buildings',
    '001_MM_L3_NON_RESIDENTIAL_BUILDING':'buildings',
    '001_MM_L5_MISC_SMALL_STRUCTURE':None,
    '002_TR_L3_GOOD_ROADS':None,
    '002_TR_L4_POOR_DIRT_CART_TRACK':None,
    '002_TR_L6_FOOTPATH_TRAIL':None,
    '006_VEG_L2_WOODLAND':None,
    '006_VEG_L3_HEDGEROWS':None,
    '006_VEG_L5_GROUP_TREES':None,
    '006_VEG_L5_STANDALONE_TREES':'trees',
    '007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND':None,
    '007_AGR_L6_ROW_CROP':None,
    '007_AGR_L7_FARM_ANIMALS_IN_FIELD':'animals',
    '008_WTR_L3_WATERWAY':None,
    '008_WTR_L2_STANDING_WATER':None,
    '003_VH_L4_LARGE_VEHICLE':'vehicle',
    '003_VH_L5_SMALL_VEHICLE':'vehicle',
    '003_VH_L6_MOTORBIKE':'vehicle'
}


class DSTLImage:
    """
    Attributes
    ----------
    _W (float) : width of the 3 ch images
    _H (float) : height of the 3 ch images
    _grid_sz (Pandas data frame) : look up of the xmax and ymax image scaling
    _image_id (str) : id of the image of the form xxxx_xx_xx
    _xmax (float) : the xmax scaling value found from grid_sz
    _ymin (float) : the ymin scaling value found from grid_sz
    _data (array : float) : image data of shape _W x _H x 3
    _features (dict : list) : a list of all polygons vertices to plot for each feature

    """

    _FLAGS = {'errors':{}}
    geojson_dir = None
    grid_sizes = None
    
    def __init__(self, fname):
        """Load all necessary info to describe a 3ch image

        Args
        ----
        fname (str) : the file name of the 3ch image to import 
        """

        self._image_id = pipe_utils.get_file_name_from_path(fname)

        # import the data for the image and the json
        print("loading image ", self._image_id)
        _data = pipe_utils.read_raw_image(fname)
        self._data = pipe_utils.image_preprocessor(_data)
        del _data
        print("...done")

        # correct the widths and heights via the "Data Processing tut" on *
        # Note: * is bullsh!t, find the image shape yourself
        _H, _W, _ = self.data.shape
        self._W = _W**2/(_W+1)
        self._H = _H**2/(_H+1)

        # save the xmax and ymin values to resize the mask correctly
        _, self._xmax, self._ymin = self.grid_sizes[
                self.grid_sizes.image == self._image_id].values[0]

        print("loading json file")
        self._features = self._parse_geojson(self.geojson_path)
        print("...done")


    @property
    def data(self):
        """Return the satellite image data 
        """
        return self._data


    @property
    def image_id(self):
        return self._image_id


    @property
    def geojson_path(self):
        """Find the correct json folder for a given image
        """
        return os.path.join(self.geojson_dir, self.image_id)


    def has_labels(self):
        """Return the labels present in this image

        Returns
        -------
        (list : str) : a list of the label names
        """
        return {_id:CLASS_ID_TO_NAME[_id] for _id in self._features.keys()}


    def _parse_geojson(self, json_dir):
        """Iterate through each label file in a geojson dir and extract the 
        feature locations

        Args
        ----
        json_dir (self) : the directory of the json files

        Returns
        -------
        dic of form {label: [ [feature coor] ]
        """

        features = {}

        # convert form dstl coors to pixels
        coor_to_px = lambda coor: (
                coor[0]/self._xmax*self._W, 
                coor[1]/self._ymin*self._H)


        # track number of errors
        e_count = 0

        import pudb; pudb.set_trace()

        # For each feature type (separate file) in the image's json directory, 
        # store the feature locations
        for fname in glob.glob(json_dir + "/*.geojson"):         

            try:
                dstl_label = pipe_utils.get_file_name_from_path(fname)
            except KeyError as e:
                print(e)
                pass
            else:
                # Load the file with geojson, this is the same as json.load()
                with open(fname) as f:
                    raw_json = geojson.load(f)

                # parse the mask (geojson polygon) for each object in a feature
                for feature in raw_json['features']:
                    try:
                        # Each Label in a file will be the same, but a different 
                        # mask will be stored for each item
                        str_label = feature['properties']['LABEL']
                    except KeyError:
                        self._FLAGS['errors']['KeyError'] = e_count
                    else:
                        coors = feature['geometry']['coordinates'][0]
                        try:
                            features.setdefault(
                                    dstl_label, 
                                    []).append(list(map(coor_to_px, coors)))
                        except TypeError:
                            e_count +=1
                            self._FLAGS['errors']['TypeError'] = e_count
        else:
            print("\nNo GeoJson file found for that image\n")


        return features


    def get_feature_name_by_id(self, label_id):
        """Return the verbose name of a feature given its int identifier

        Note
        ----
        Should just make this a global method, doesn't need to be in the class

        Args
        ----
        label_id (int) : the int identifier of a feature

        Returns
        -------
        (str) : verbose name of feature
        """
        return CLASS_ID_TO_NAME[label_id]


    def get_feature_locations(self, label_id=None, as_bbox=True):
        """Return a list of all the feature locations

        Args
        ----
        label_id (int) : the int identifier of a feature
        as_bbox (bool) : (False) return obj location as a bounding box instead
        of a polygon

        Returns
        -------
        list of arrays, for element in the list is a feature location, the 
        corresponding array is the coor of the polygon (in px)

        if as_bbox : list is of the form [x,y,w,h]
        """

        locations = self._features[label_id]

        if as_bbox:
            _temp_locs = []
            for loc in locations:
                xloc, yloc = list(zip(*loc))
                xmin = np.min(xloc)
                xmax = np.max(xloc)
                ymin = np.min(yloc)
                ymax = np.max(yloc)
                _temp_locs.append([xmin, ymin, xmax, ymax])

            locations = _temp_locs

        return locations

            
    def show(self, label_ids=[], colors=['r'], as_bbox=False):
        """Display the satellite image with the feature overlay

        Args
        ----
        label_ids (list) : list of ints corresponding to feature ids
        colors (list : str) : currently not used
        as_bbox (bool) : id the object locations as a box instead of polygon
        """

        plt.imshow(self.data)
        ax = plt.gca()
        
        for label_id in label_ids:

            name = CLASS_ID_TO_NAME[label_id]

            for loc in self.get_feature_locations(label_id=label_id, as_bbox=as_bbox):
                if as_bbox:
                    x1, y1, x2, y2 = loc
                    xy = (x1, x2)
                    width = x2-x1
                    height = y2-y1
                    patch = patches.Rectangle(
                                xy=xy, width=width, height=height,
                                label=name, 
                                color=colors[0],
                                fill=False
                            )
                else:
                    patch = patches.Polygon(
                                loc, 
                                label=name, 
                                color=colors[0],
                                alpha=.2
                            )

                ax.add_patch(patch)

        plt.legend()


def get_dstl_feature_label_from_path(path):
    """Get the feature label from the path to  
    """



def dstl_loader(geojson_dir, grid_sizes):
    """Create a closure to more easily call DSTL Image with the image_dir and 
    geojson dir

    Args
    ----
    geojson_dir (str) : path to the directory containing the geojson files
    """

    DSTLImage.geojson_dir = geojson_dir
    DSTLImage.grid_sizes = grid_sizes

    def _dstl_image(path):

        return DSTLImage(path)

    return _dstl_image


def import_grid_sizes(path):
    """Pull in the grid_size table as a pandas dataframe

    This file is necessary to map the bbox coordinates to pixels. 

    "To utilize these images, we provide the grid coordinates of each image so 
    you know how to scale them and align them with the images in pixels. You 
    need the Xmax and Ymin for each image to do the scaling (provided in our 
    grid_sizes.csv)" - (*)

    Dataframe columns: image, Xmax, Ymax

    Args
    ----
    path (str) : file location of grid_sizes.csv, provided by (*)

    Returns
    ------
    Pandas data frame
    """
    df = pd.read_csv(path)
    df.rename(columns={'Unnamed: 0':'image'}, inplace=True)
    print("Loaded grid_sizes")
    print(df.head())
    return df


def geojson_to_bbox():
    pass


if __name__ == '__main__':
    grid_file = "/Users/bdhammel/Documents/insight/data/dstl/grid_sizes.csv"
    geojson_dir = "/Users/bdhammel/Documents/insight/data/dstl/train_geojson_v3"
    img_path = "/Users/bdhammel/Documents/insight/data/dstl/three_band/6010_1_2.tif"
    grid_sizes = import_grid_sizes(grid_file)

    loader = dstl_loader(geojson_dir=geojson_dir, grid_sizes=grid_sizes)

    img = loader(img_path)

