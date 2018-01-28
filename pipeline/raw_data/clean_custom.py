"""This script handles cleaning data that was labeled by Harvesting 

Images consist of labeled images from UC Merced's sparse residential category

References
----------
 (*) http://weegee.vision.ucmerced.edu/datasets/landuse.html

"""
import glob
import os
import csv
import skimage.io as skio
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np 
from pipeline import utils as pipe_utils


MAP_TO_LOCAL_LABELS = {
    'building':'buildings',
    'vehicle':'vehicle'
    'waterbody':None
}

class HarvestingImage:

    def __init__(self, img_file):
        self._path = img_file
        _data = pipe_utils.read_raw_image(img_file, report=False)
        self._data = pipe_utils.image_preprocessor(_data)
        self._features = {}


    @property
    def path(self):
        return self._path


    @property
    def image_name(self):
        return pipe_utils.get_file_name_from_path(self.path)

    @property
    def data(self):
        return self._data

    @property
    def features(self):
        return self._features

    def append_feature(self, label, coor):
        self._features.setdefault(label, []).append(coor)

    def show(self, label):

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


def load_images(image_dir, xml_annotations_dir):
    """Read in from the XML directory output by RetcLabel
    """

    ds = []

    for annotation_file in glob.glob(xml_annotations_dir + '*.xml'):
        tree = ET.parse(annotation_file)
        r = tree.getroot()
        img_file = os.path.join(image_dir, r.find('filename').text)

        img = HarvestingImage(img_file)

        for obj in r.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            img.append_feature(label, (xmin,ymin,xmax,ymax))

        ds.append(img)

    return ds


def process_directory(dir_path, image_save_dir, annotations_save_dir):
    """

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
    xml_annotation_dir = os.path.join(image_dir, 'annotations/')

    ds = load_images(dir_path, xml_annotation_dir)

    with open(annotations_save_path, 'a') as csv_file:

        annotation_writer = csv.writer(csv_file)
        
        for img in ds:
            img_path = os.path.join(image_save_dir, img.image_name+".png")
            skio.imsave(
                    img_path,
                    img.data
            )

            for label, features in img.features.items():
                label = MAP_TO_LOCAL_LABEL[label]
                for coor in features:
                    row = [img_path, *coor, label]

                    # output to file
                    annotation_writer.writerow(row)


if __name__ == '__main__':

    image_dir = '/Users/bdhammel/Documents/insight/data/harvesting'
    image_save_dir = '/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/harvesting/images/'
    annotations_save_dir = '/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/harvesting/'
    process_directory(
            dir_path=image_dir,
            image_save_dir=image_save_dir,
            annotations_save_dir=annotations_save_dir
    )

