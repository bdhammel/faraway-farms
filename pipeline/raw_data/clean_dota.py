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
from skimage.util import view_as_blocks
import skimage.io as skio
import numpy as np 
import glob
import os
import csv

from pipeline import utils as pipe_utils
import pipeline.raw_data.utils as clean_utils
from pipeline import obj_pipeline


MAP_TO_LOCAL_LABELS = {
    'plane', 
    'ship', 
    'storage-tank', 
    'baseball-diamond', 
    'tennis-court', 
    'basketball-court', 
    'ground-track-field', 
    'harbor', 
    'bridge', 
    'large-vehicle', 
    'small-vehicle', 
    'helicopter', 
    'roundabout', 
    'soccer-ball-field',
    'basketball-court',
    'swimming-pool'
}


class DOTAImage(clean_utils.RawObjImage):


    def get_features(self, as_poly=False):
        """Return featues included in the image

        defaults to returning features as bounding boxes 

        Args
        ----
        as_poly (bool) : return the feature location in the original format,  
            usually as a bounding polygon 
        """

        if as_poly: 
            features = self.features
        else:
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

        img.check_self()

        return img
    
    return _loader



def dota_processor(block_shape):
    """
    """

    ystride, xstride, *_ = block_shape

    def __transform_coors(coor, i, j):
        """Map coordinates into the new space

        Args
        ----
        coor
        i
        j
        """
        x1, y1, x2, y2 = coor
        xcenter = (x2+x1)/2
        ycenter = (y2+y1)/2
        dx = x2 - x1
        dy = y2 - y1
        if xcenter // xstride == i and ycenter // ystride == j:
            newx = xcenter % xstride
            newy = ycenter % ystride
            x1_prime = int(np.maximum(newx - dx/2, 0))
            y1_prime = int(np.maximum(newy - dy/2, 0))
            x2_prime = int(np.minimum(newx + dx/2, xstride))
            y2_prime = int(np.minimum(newy + dy/2, ystride))

            return (x1_prime, y1_prime, x2_prime, y2_prime)


    def _processor(raw_img):

        blocks = pipe_utils.chop_to_blocks(raw_img.data, block_shape)
        jmx, imx, *_ = blocks.shape

        features = raw_img.get_features()

        ds = []

        # save each block as it's own obj image, and convert features to the 
        # new coordinate ref frame
        for j in range(jmx):
            for i in range(imx):
                img_patch = obj_pipeline.ObjImage(data=blocks[j, i, 0, ...])
                new_id = raw_img.image_id + "__{}_{}".format(i, j)
                img_patch.set_image_id(new_id)
                for label, locs in features.items():
                    for loc in locs:
                        loc = __transform_coors(loc, i, j)
                        if loc:
                            img_patch.append_feature(label, loc)

                ds.append(img_patch)

        return ds
    
    return _processor


def save_as_retinanet_data(
        ds, 
        image_save_dir, 
        annotations_save_dir, 
        percent_test_set=.2
):
    """Save the data in the csv format expected by RetinaNet

    Args
    ----
    ds (list : ObjImage) : Image data set
    image_save_dir (str) :
    annotations_save_dir (str) :
    percent_test_set (float) : value in [0, 1), percent of data to use as test
    """
    

    test_annot_path = os.path.join(annotations_save_dir, "test_annot.csv")
    train_annot_path = os.path.join(annotations_save_dir, "train_annot.csv")

    if not os.path.exists(image_save_dir):
        print("Creating directory to save processed images")
        os.makedirs(image_save_dir)

    if not os.path.exists(annotations_save_dir):
        print("Creating directory to save annotation file")
        os.makedirs(annotations_save_dir)


    with open(test_annot_path, 'a') as test_file, open(train_annot_path, 'a') as train_file:

        test_writer = csv.writer(test_file)
        train_writer = csv.writer(train_file)

        for img in ds:
            image_path = os.path.join(image_save_dir, img.image_id+'.png')
            skio.imsave(image_path, img.data)

            # Randomly determine if the image is for training or testing 
            if np.random.uniform() < .2:
                writer = test_writer
            else:
                writer = train_writer

            # Make a note of every feature and bounding location in the image
            all_features = img.get_features()
            for label, locations in all_features.items():
                for location in locations:
                    row = [image_path, *location, label]
                    writer.writerow(row)

            # Write a blank row if no features in this image
            if not all_features:
                row = [image_path, '', '', '', '', '']
                writer.writerow(row)


if __name__ == "__main__":
    pass
    #loader = dota_loader('/Users/bdhammel/Documents/insight/data/dota/labelTxt')
    #raw_img = loader('/Users/bdhammel/Documents/insight/data/dota/images/P1872.png')
    ##print(raw_img.has_labels())
    ##img.show('plane', as_bbox=False)
    #processor = dota_processor(block_shape=(400,400,3))
    #ds = processor(raw_img)

    #save_as_retinanet_data(
    #    ds, 
    #    '/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/dota/images',
    #    '/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/dota/'
    #)






