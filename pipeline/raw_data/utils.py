import numpy as np
from PIL import Image, ImageDraw

from pipeline import utils as pipe_utils


class RawPatchImage(pipe_utils.SatelliteImage):
    pass


class RawObjImage:


    def __init__(self, image_path):
        self._image_id = pipe_utils.get_file_name_from_path(image_path)

        # import the data for the image and the json
        print("loading image ", self._image_id)
        _data = pipe_utils.read_raw_image(image_path)
        self._data = pipe_utils.image_preprocessor(_data)
        del _data
        print("...done")
        self._features = {}


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

    
    def check_self(self):
        datamax = self.data.max()
        if datamax < 1 and self.data.min() > 0:
            print("[0,1)")
        elif datamax < 255:
            print("[0,255]")


    def append_feature(self, label, coor):
        """Connect a feature to the imported image

        Args
        ----
        label (str) : label of the feature 
        """
        self._features.setdefault(label, []).append(coor)


    def has_labels(self):
        """Return the labels present in this image

        Returns
        -------
        (list : str) : a list of the label names
        """
        return list(self.features.keys())


    def show(self, labels=[], as_bbox=True):
        """Display the satellite image with the feature overlay

        Args
        ----
        labels (list) : list of ints corresponding to feature names
        """

        labels = pipe_utils.atleast_list(labels)

        im = Image.fromarray(self.data)
        draw = ImageDraw.Draw(im)       

        for label in labels:
            locs = self.get_features(as_bbox=as_bbox)[label]

            for coors in locs:
                if as_bbox:
                    draw.rectangle(
                            coors,
                            outline='red'
                    )
                else:
                    draw.polygon(
                            coors,
                            outline='red'
                    )

        im.show()




