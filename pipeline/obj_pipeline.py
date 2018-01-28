import csv
import skimage.io as skio
import matplotlib.pyplot as plt
from matplotlib import patches
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
            self._data = skio.imread(image_path)
            self._image_id = pipe_utils.get_file_name_from_path(image_path)
        elif data is not None:
            self._data = data
            self._image_id = None

        self._check_data(self._data)
        self._features = {}


    def set_image_id(self, image_id):
        """Give this data an id

        Typically this is just the file name

        Args
        ----
        image_id (str) : the id to assign the image
        """
        self._image_id = image_id


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


    def show(self, label = None):
        """Display the image

        Args
        ----
        label (str) : label to be plotted with the image
        """

        plt.imshow(self._data)
        ax = plt.gca()

        if label is not None:
            features = self._features.get(label, [])
        else:
            features = self.get_features()

        for loc in features:
            x1, y1, x2, y2 = loc
            xy = (x1, y1)
            width = x2-x1
            height = y2-y1
            patch = patches.Rectangle(
                        xy=xy, width=width, height=height,
                        label=label, 
                        color='r',
                        fill=False
                    )

            ax.add_patch(patch)


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



if __name__ == '__main__':
    ds = load_data('/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/harvesting/annotations.csv')
    plt.ion()

    for data in ds:
        plt.close('all')
        if 'building' in data.has_labels():
            data.show('building')
            input("press enter to continue")

