"""Functions specific to handling cleaned data for the Object detection model
"""

import csv
import skimage.io as skio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import os

from pipeline import utils as pipe_utils


class ObjImage(pipe_utils.SatelliteImage):
    """Class to handle images with bounding box attributes

    ObjImage inherits from Satellite Image to include the generic properties:

    Parameters
    ----------
    _data (np.array, int) : image data of the type uint8 with range [0,255]
    _image_id (str) : a unique identifier for the image, if the image is saves
    this becomes the file name
    _features ( {label: [(int, int, int, int), ...], ...}) : (optional) 
        object features in the image, this property isn't used with patch images
    """


    def __init__(self, image_path=None, data=None, *args, **kwargs):
        """Loads an image from a processed directory 

        Args
        ----
        image_path (str) : location of the image to upload
        data (np.array) : raw data to load. Data is loaded this way in the  
        last stage of cleaning, to ensure that internal tests pass
        """

        if image_path is not None:
            _data = pipe_utils.read_raw_image(image_path)
            _image_id = pipe_utils.get_file_name_from_path(image_path)
        elif data is not None:
            _data = data
            _image_id = None

        super().__init__(data=_data, image_id=_image_id, use='obj', *args, **kwargs)


    def get_features(self):
        """Return all the features associated with the image

        Returns 
        -------
        dict {label : [(x1, y1, x2, y2), (x1, y1, x2, y2), ...], ...}
        """
        return self._features


    def append_feature(self, label, coor):
        """Connect a feature to the imported image

        Args
        ----
        label (str) : label of the feature 
        coor (tuple : int) : coordinates of the bbox, of form (x1, y1, x2, y2)
        """
        self._features.setdefault(label, []).append(coor)


    def has_labels(self):
        """Return the labels connected to this image

        Returns
        --------
        [label1, label2, ...]
        """
        return list(self._features.keys())


    def show(self, labels=None, return_image=False):
        """Display the image

        Args
        ----
        labels (str) : label to be plotted with the image, can be str, list, 
            or 'all' to plot all labels in the image
        return_image (bool) : don't plot the image, just return it, used for 
            plotting inline with jupyter notebooks
        """

        im = Image.fromarray(self.data)

        # If 'all' get all labels in the image
        if labels == 'all':
            labels = self.has_labels()

        # Make sure labels is a list
        labels = pipe_utils.atleast_list(labels)

        # Initialize a drawer, and draw each feature
        draw = ImageDraw.Draw(im)       

        for label in labels:
            locs = self.get_features()[label]

            for coors in locs:
                draw.rectangle(
                        coors,
                        outline='red'
                )

        if return_image:
            return im
        else:
            im.show()


def load_data(annotations_file, max_images=100):
    """Load data in from a CSV annotations file

    Args
    ----
    annotations_file (str) : the path to the annotations.csv file
    max_images (int) : the maximum number of images to load from a given 
    annotations_file
    """

    dataset = {}

    with open(annotations_file) as f:
        csv_reader = csv.reader(f)

        for img_path, *coor, label in csv_reader:

            # Convert coor to ints (pixels), if no coor, then just pass
            try:
                coor = tuple(map(float, coor))
            except ValueError as e:
                pass
            else:
                img = dataset.get(img_path, ObjImage(img_path))
                img.append_feature(label, coor)
                dataset[img_path] = img

            if len(dataset.keys()) >= max_images: 
                break

    return list(dataset.values())


def update_annotation_file_img_paths(annotation_file_path, new_img_dir):
    """Automated updating of the image paths in an annotation file
    If the dataset directory gets moved. e.g. to a server
    the image paths in the annotation files will need to be updated

    Args
    ----
    annotation_file_path (str) : path to the current annotation file
    new_img_dir (str) : the new directory where the images are stored
    """
    
    annotation_dir = os.path.dirname(annotation_file_path)
    annotation_file = os.path.basename(annotation_file_path)
    old_annotation_file_path = os.path.join(annotation_dir, "old_"+annotation_file)
    os.rename(annotation_file_path, old_annotation_file_path)

    with open(old_annotation_file_path, 'r') as old_file, open(annotation_file_path, 'w') as new_file:
        old_reader = csv.reader(old_file)
        new_writer = csv.writer(new_file)

        # update each image location in the annotations file
        for row in old_reader:
            old_img_path = row[0]
            image_file_name = os.path.basename(old_img_path)
            new_img_path = os.path.join(new_img_dir, image_file_name)
            new_writer.writerow([new_img_path, *row[1:]])


def merge_annotation_files(file1, file2, target_file):
    """
    Args
    ----
    file1 (str) : path 
    file2 (str) : path
    target_file (str) : path
    """

    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(target_file, 'w') as tf:
        file1_reader = csv.reader(f1)
        file2_reader = csv.reader(f2)
        target_writer = csv.writer(tf)

        for row in file1_reader:
            target_writer.writerow(row)

        for row in file2_reader:
            target_writer.writerow(row)




if __name__ == '__main__':
    ds = load_data('/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/harvesting/annotations.csv')

    for data in ds:
        if 'building' in data.has_labels():
            data.show(labels='building')
            input("press enter to continue")

