"""This script handles pulling in the data from the kaggle competition from DSTL, 
cleaning it, and saving it in a format that's appropriate to be read in by the 
keras-retinanet model


References
----------
(*) https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection

"""
import matplotlib.pyplot as plt
from matplotlib import patches
import skimage.io as skio
from skimage.util import view_as_blocks
import numpy as np 
import glob
import geojson
import pandas as pd
import os
import csv

from pipeline import utils as pipe_utils
import pipeline.raw_data.utils as clean_utils


MAP_TO_LOCAL_LABELS = {
    'LARGE_BUILDING':'buildings',
    'RESIDENTIAL_BUILDING':'buildings',
    'NON_RESIDENTIAL_BUILDING':'buildings',
    'MISC_SMALL_MANMADE_STRUCTURE':'buildings',
    'GOOD_ROADS':None,
    'POOR_DIRT_CART_TRACK':None,
    'FOOTPATH_TRAIL':None,
    'WOODLAND':None,
    'HEDGEROWS':None,
    'GROUP_TREES':None,
    'STANDALONE_TREES':'trees',
    'CONTOUR_PLOUGHING_CROPLAND':None,
    'ROW_CROP':None,
    'DEMARCATED_NON_CROP_FIELD':None,
    'FARM_ANIMALS_IN_FIELD':'animals',
    'WATERWAY':None,
    'STANDING_WATER':None,
    'LARGE_VEHICLE':'vehicle',
    'SMALL_VEHICLE':'vehicle',
    'MOTORBIKE':'vehicle'
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
    def features(self):
        return self._features


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
        return {label:pipe_utils.OBJ_CLASS_TO_ID[label] 
                for label in self._features.keys()}


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

        # For each feature type (separate file) in the image's json directory, 
        # store the feature locations
        for fname in glob.glob(json_dir + "/*.geojson"):         

            # Load the file with geojson, this is the same as json.load()
            with open(fname) as f:
                raw_json = geojson.load(f)

            # parse the mask (geojson polygon) for each object in a feature
            for feature in raw_json['features']:
                try:
                    dstl_label = feature['properties']['LABEL']
                except KeyError:
                    self._FLAGS['errors']['KeyError'] = e_count
                else:
                    label = MAP_TO_LOCAL_LABELS[dstl_label]
                    if label:
                        coors = feature['geometry']['coordinates'][0]
                        try:
                            features.setdefault(
                                    label, 
                                    []).append(list(map(coor_to_px, coors)))
                        except TypeError:
                            e_count +=1
                            self._FLAGS['errors']['TypeError'] = e_count

        else:
            # There were no files in that json directory
            if not features:
                print("No file found")

        return features


    def get_feature_locations(self, label=None, as_bbox=True):
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

        locations = self._features[label]

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

            
    def show(self, labels=[], colors=['r'], as_bbox=False):
        """Display the satellite image with the feature overlay

        Args
        ----
        label_ids (list) : list of ints corresponding to feature ids
        colors (list : str) : currently not used
        as_bbox (bool) : id the object locations as a box instead of polygon
        """

        plt.imshow(self.data)
        ax = plt.gca()
        
        for label in labels:

            for loc in self.get_feature_locations(label=label, as_bbox=as_bbox):
                if as_bbox:
                    x1, y1, x2, y2 = loc
                    xy = (x1, y1)
                    width = x2-x1
                    height = y2-y1
                    patch = patches.Rectangle(
                                xy=xy, width=width, height=height,
                                label=label, 
                                color=colors[0],
                                fill=False
                            )
                else:
                    patch = patches.Polygon(
                                loc, 
                                label=label, 
                                color=colors[0],
                                alpha=.2
                            )

                ax.add_patch(patch)

        plt.legend()



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



def __transform_and_collect_features(dstl_image, stride, blocks_shape):
    """Returns the features transformed into the new cor system

    Returns
    -------
    { block_tag : {label : [ [x1,y1,x2,y2], ... ] }, ... }
    with block_tag = "i_j"
    """
    # get the x, y, stride, ignore color channel it should be 3
    ystride, xstride, _ = stride 
    base_id = dstl_image.image_id
    jmx, imx, *_ = blocks_shape

    # temp sanity check
    assert ystride == xstride
    
    collection = {}
    ignored_fet = 0

    for label in dstl_image.has_labels():
        for feature in dstl_image.get_feature_locations(label=label, as_bbox=True):

            x1, y1, x2, y2 = feature
            x_center = (x2+x1)/2
            y_center = (y2+y1)/2
            dx = x2-x1
            dy = y2-y1
            assert x_center > 0
            assert y_center > 0
            assert dx > 0
            assert dy > 0
            
            iblock = int(x_center // xstride)
            jblock = int(y_center // ystride)

            # Only include a feature if its with
            if iblock <= imx and jblock <= jmx:
                block_tag = "{}_{}".format(iblock, jblock)
                x1_prime = int(x1 % xstride)
                y1_prime = int(y1 % ystride)
                x2_prime = int(x2 % xstride)
                y2_prime = int(y2 % ystride)

                collection.setdefault(block_tag, {})
                collection[block_tag].setdefault(label, [])
                collection[block_tag][label].append(
                        [x1_prime, y1_prime, x2_prime, y2_prime]
                )

            else:
                ignored_fet += 1
                print('{0} features ignored'.format(ignored_fet), end="\r")

    return collection


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


def process_dstl_directory(dir_path, image_save_dir, annotations_save_dir):
    """For a directory of DSTL images, import and process into acceptable model
    format

    Generate a directory of chunked images. i.e. split larget images into smaller
    blocks, transform the features into the new pixel coordinates, and save into 
    and annotations file to be read by retinanet 

    Args
    ----
    dir_path
    image_save_path
    annotation_save_path
    """

    if not os.path.exists(image_save_dir):
        print("Creating directory to save processed images")
        os.makedirs(image_save_dir)

    if not os.path.exists(annotations_save_dir):
        print("Creating directory to save annotation file")
        os.makedirs(annotations_save_dir)

    annotations_save_path = os.path.join(annotations_save_dir, 'annotations.csv')
    loader = dstl_loader(geojson_dir=geojson_dir, grid_sizes=grid_sizes)

    with open(annotations_save_path, 'a') as csv_file:

        csvwriter = csv.writer(csv_file)
        processor = dstl_processor(
                block_shape=(300,300,3),
                image_save_dir=image_save_dir,
                annotations_writer=csv_writer
        )


        for img_path in glob.iglob(dir_path + '*.tif'):
            print("processing image: ", img_path)
            dstl_image = loader(img_path)
            processor(dstl_image)

 

def dstl_processor(block_shape, image_save_dir, annotation_writer):
    """Process the dstl images and save them in a format that can be read in 
    by the model

    The DSTL images are huge, cut them down to bite sized chunks and 

    Args
    ----
    dstl_image (DSTLImage)
    img_save_path (str) : path to directory to save the image
    annotaton_writer (csv.writer) : a file writer for csv
    """


    def _process(dstl_image):

        # Make sure there are not multiple strides in the color ch direction 
        assert block_shape[-1] == dstl_image.data.shape[-1]

        # Drop parts of the image that cant be captured by an integer number of 
        # strides 
        _split_factor = np.floor(
                np.divide(dstl_image.data.shape, block_shape)).astype(int)
        _img_lims = (_split_factor * block_shape)

        print("Can only perserve up to pix: ", _img_lims)

        _data = np.ascontiguousarray(
                dstl_image._data[:_img_lims[0], :_img_lims[1], :_img_lims[2]]
                )

        blocks = view_as_blocks(_data, block_shape) 

        ystride, xstride, _ = block_shape
        jmx, imx, *_ = blocks.shape

        # Transform the features of an image into the coordinates of the new
        # blocks
        collection = __transform_and_collect_features(
                dstl_image,
                block_shape, 
                blocks.shape)


        for j in range(jmx):
            for i in range(imx):
                block_tag = "{}_{}".format(i, j)
                block_id = "{}__{}".format(dstl_image.image_id, block_tag)
                
                # Save the image
                block_img_path = os.path.join(image_save_dir, block_id+'.png')
                skio.imsave(block_img_path, blocks[j, i, 0, ...])

                # Save annotations
                # write the image path and x,y coor of bounding boxes to 
                # the csv file 
                # has format:
                #   path_to_image, x1, y1, x2, y2, label
                try:
                    block_features = collection[block_tag]
                except:
                    # if no features in an image, just append blank position
                    row = [block_img_path, '', '', '', '', '']
                    annotation_writer.writerow(row)
                else:
                    # for each features, append its locations and label of
                    # the features
                    for label, features in block_features.items():
                        for coor in features:
                            row = [block_img_path, *coor, label]

                            # output to file
                            annotation_writer.writerow(row)


    return _process


if __name__ == '__main__':
    grid_file = "/Users/bdhammel/Documents/insight/data/dstl/grid_sizes.csv"
    geojson_dir = "/Users/bdhammel/Documents/insight/data/dstl/train_geojson_v3"
    img_path = "/Users/bdhammel/Documents/insight/data/dstl/three_band/6010_1_2.tif"
    grid_sizes = import_grid_sizes(grid_file)

    # process_dstl_directory(
    #     '/Users/bdhammel/Documents/insight/data/dstl/three_band/',    
    #     '/Users/bdhammel/Documents/insight/harvesting/dataset/obj_detection/dstl/images',    
    #     '/Users/bdhammel/Documents/insight/harvesting/dataset/obj_detection/dstl/',    
    # )

    loader = dstl_loader(geojson_dir=geojson_dir, grid_sizes=grid_sizes)
    img = loader(img_path)
    annotations_save_dir = '/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/dstl/'
    annotations_save_path = os.path.join(annotations_save_dir, 'annotations.csv')

    # with open(annotations_save_path, 'a') as csv_file:

    #     csvwriter = csv.writer(csv_file)
    #     processor = dstl_processor(
    #             (300, 300, 3), 
    #             '/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/dstl/images',
    #             csvwriter
    #     )

    #     processor(img)


    plt.close('all')
    img.show(labels=['buildings'], as_bbox=True)
    img.show(labels=['buildings'], as_bbox=False)
    #plt.ylim(2650, 2850)
    #plt.xlim(3200, 3500)

