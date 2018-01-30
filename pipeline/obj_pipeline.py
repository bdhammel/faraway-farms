import csv
import skimage.io as skio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import os

from pipeline import utils as pipe_utils


class ObjImage(pipe_utils.SatelliteImage):


    def __init__(self, image_path=None, data=None):
        """

        Loads an image from a processed directory 
        Args
        ----
        image_path (str) : location of the image to upload
        """

        if image_path is not None:
            self._data = pipe_utils.read_raw_image(image_path)
            self._image_id = pipe_utils.get_file_name_from_path(image_path)
        elif data is not None:
            self._data = data
            self._image_id = None

        self._check_data(self._data)
        self._features = {}


    def get_features(self):
        return self._features


    def _check_data(self, data):
        #assert data.max() < 1
        #assert data.std() > .2
        pass


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


def retinanet_preprocessor(data):
    """Process data in the manner expected by retinanet

    Convert RGB -> BGR
    normalize in the VGG16 way

    Notes
    -----
     - handles batch or single image
     - do NOT use this with Retina net built in pre processor

    Args
    ----
    data (np.array) : of shape ( _, _, 3)

    Returns
    -------
    normalized data of the same shape 
    """

    # flip to BGR channel, cause that's what retina net says to do
    data = data[...,::-1]

    data[...,0] -= 103.939
    data[...,1] -= 110.779
    data[...,2] -= 123.78

    return data




if __name__ == '__main__':
    ds = load_data('/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/harvesting/annotations.csv')

    for data in ds:
        if 'building' in data.has_labels():
            data.show(labels='building')
            input("press enter to continue")

