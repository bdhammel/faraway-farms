"""This script handles cleaning data that was labeled using the program RectLabel


References
----------
 (*) https://rectlabel.com/

"""
import glob
import os
import csv
import skimage.io as skio
import xml.etree.ElementTree as ET
import numpy as np 
from skimage.transform import resize as sk_resize 
from pipeline.raw_data import utils as clean_utils
from pipeline import obj_pipeline
from skimage.color import grey2rgb


# Convert labels labeled by Harvesting
MAP_TO_LOCAL_LABELS = {
    'building':'house',         # Match with DOTA data set 
    'vehicle':'vehicle',
    'waterbody':None
}



class RectLabelImage(clean_utils.RawObjImage):
    """Class of a 
    """

    def get_features(self, *args, **kwargs):
        return self._features


def load_raw_images(image_dir, xml_annotations_dir):
    """Read in from the XML directory output by RetcLabel

    Reads in the images and parses the xml
    Labels read in are not updated to local labels 

    Args
    ----

    Returns
    -------
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

def __process_custom(data):

    # drop alpha channel
    # make sure RGB channels
    if np.ndim(data.shape) == 3:
        if data.shape[-1] == 4:
            data = data[...,:3]
    else:
        data = grey2rgb(data)
    
    # make sure data is [0, 255]
    if data.max() < 1 and data.min() > 0:
        data *= 255

    data = data.astype(np.uint8)

    return data


def process_raw_dataset(raw_ds, target_size=(400,400)):
    ds = []
    for raw_img in raw_ds:

        yshape, xshape, _ = raw_img.data.shape
        yratio = target_size[0]/yshape
        xratio = target_size[1]/xshape
        data = sk_resize(raw_img.data, target_size, preserve_range=True)
        data = __process_custom(data)

        img = obj_pipeline.ObjImage(data=data)
        img.set_image_id(raw_img.image_id)

        for label, features in raw_img.get_features().items():
            local_label = MAP_TO_LOCAL_LABELS[label]
            if local_label is not None:
                for feature in features:
                    x1, y1, x2, y2 = feature
                    # if box was drawn from the bottom right to the top left, 
                    # switch the order
                    if (x1 > x2) and (y1 > y2):
                        _x1 = x1
                        _y1 = y1
                        x1 = x2
                        y1 = y2
                        x2 = _x1
                        y2 = _y2

                    img.append_feature(
                            local_label, 
                            [
                                int(float(x1*xratio)), 
                                int(float(y1*yratio)), 
                                int(float(x2*xratio)), 
                                int(float(y2*yratio))
                            ]
                    )

        ds.append(img)

    return ds



def augment(img):
    return img



def data_generator(ds, max_num):

    for i in range(max_num):
        np.random.seed()
        random_img = np.random.choice(ds)
        _id = int(1e5*np.random.random())
        random_img.set_image_id(random_img.image_id + '_{}'.format(_id))
        random_img = augment(random_img)
        
        yield random_img


def save_dataset(ds, upsample=False, max_images=100):
    if not os.path.exists(image_save_dir):
        print("Creating directory to save processed images")
        os.makedirs(image_save_dir)

    if not os.path.exists(annotations_save_dir):
        print("Creating directory to save annotation file")
        os.makedirs(annotations_save_dir)

    annotations_save_path = os.path.join(annotations_save_dir, 'annotations.csv')

    with open(annotations_save_path, 'a') as csv_file:

        annotation_writer = csv.writer(csv_file)

        if upsample:
            ds_gen = data_generator(ds, max_images)
        
        for img in ds_gen:
            img_path = os.path.join(image_save_dir, img.image_id+".png")
            skio.imsave(
                    img_path,
                    img.data
            )

            for label, features in img.get_features().items():
                for coor in features:
                    row = [img_path, *coor, label]

                    # output to file
                    annotation_writer.writerow(row)



def process_directory(image_dir, image_save_dir, annotations_save_dir):
    """Read in and process a full directory

    Saves the processed information

    Args
    ----
    image_dir
    image_save_path
    annotation_save_path
    """
    pass



if __name__ == '__main__':

    image_dir = '/Users/bdhammel/Documents/insight/data/harvesting'
    image_save_dir = '/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/harvesting/images/'
    annotations_save_dir = '/Users/bdhammel/Documents/insight/harvesting/datasets/obj_detection/harvesting/'
    # process_directory(
    #         image_dir=image_dir,
    #         image_save_dir=image_save_dir,
    #         annotations_save_dir=annotations_save_dir
    # )


    xml_annotation_dir = os.path.join(image_dir, 'annotations/')

    raw_ds = load_raw_images(image_dir, xml_annotation_dir)
    ds = process_raw_dataset(raw_ds)
    save_dataset(ds, upsample=True, max_images=1000)

