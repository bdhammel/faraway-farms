"""

'imagesource'(from GoogleEarth, GF-2 or JL-1)
’gsd’(ground sample distance, the physical size of one image pixel, in meters)

Annotation format
'imagesource':imagesource
'gsd':gsd
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult

References
----------
(*) http://captain.whu.edu.cn/DOTAweb/

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

from PIL import Image, ImageDraw

from pipeline import utils as pipe_utils
import utils as clean_utils


MAP_TO_LOCAL_LABELS = {
    'plane', 
    'ship', 
    'storage tank', 
    'baseball diamond', 
    'tennis court', 
    'basketball court', 
    'ground track field', 
    'harbor', 
    'bridge', 
    'large vehicle', 
    'small vehicle', 
    'helicopter', 
    'roundabout', 
    'soccer ball field',
    'basketball court'
}


class DOTAImage(clean_utils.RawObjImage):


    def __init__(self, fname, *args, **kwargs):
        super().__init__(fname, *args, **kwargs)

   
    def get_features(self, as_bbox=True):

        if as_bbox: 
            features = {}

            for label, coors in self.features.items():
                for x1, y1, x2, y2, x3, y3, x4, y4 in coors:
                    xmax = np.max([x1, x2, x3, x4])
                    xmin = np.min([x1, x2, x3, x4])
                    ymax = np.max([y1, y2, y3, y4])
                    ymin = np.min([y1, y2, y3, y4])

                    features.setdefault(label, []).append(
                            (xmin, ymin, xmax, ymax)
                    )
        else:
            features = self.features

        return features



def dota_loader(labels_dir):

    def _loader(img_file):

        img = DOTAImage(img_file)
        label_file = os.path.join(labels_dir, img.image_id+'.txt')

        with open(label_file) as f:
            img_src = f.readline()
            gsd = f.readline()
            for line in f:
                *coor, label, _ = line.split()

                coor = tuple(map(int, coor))

                img.append_feature(label, coor)

        return img
    
    return _loader





def dota_processor(block_shape, image_save_dir, annotation_writer):
    pass






if __name__ == "__main__":
    plt.close('all')
    loader = dota_loader('/Users/bdhammel/Documents/insight/data/dota/labelTxt')
    img = loader('/Users/bdhammel/Documents/insight/data/dota/images/P1872.png')
    print(img.has_labels())
    img.show('plane', as_bbox=False)
