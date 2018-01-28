import csv
import skimage.io as skio
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from pipeline import utils as pipe_utils


class ObjImage:

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


    @property
    def data(self):
        return self._data


    @property
    def image_id(self):
        if self._image_id is None:
            raise Exception("No id for this image")

        return self._image_id

    
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


    def show(self, label):
        """Display the image

        Args
        ----
        label (str) : label to be plotted with the image
        """

        plt.imshow(self._data)
        ax = plt.gca()

        for loc in self._features.get(label, []):
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
        plt.legend()


def load_data(annotations_file):
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
                coor = tuple(map(int, coor))
            except ValueError:
                pass
            else:
                img = dataset.get(img_path, ObjImage(img_path))
                img.append_feature(label, coor)

            dataset[img_path] = img

    return list(dataset.values())


if __name__ == '__main__':
    ds = load_data('/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/harvesting/annotations.csv')
    plt.ion()

    for data in ds:
        plt.close('all')
        if 'building' in data.has_labels():
            data.show('building')
            input("press enter to continue")
