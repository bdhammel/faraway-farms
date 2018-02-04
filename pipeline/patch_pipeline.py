"""Functions specific to handling cleaned data for the patch detection model
"""
import numpy as np
from pipeline import utils as pipe_utils


class PatchDataSet:

    class_to_id = pipe_utils.PATCH_CLASS_TO_ID

    def __init__(self, path):
        """Load pickled data into python

        TODO
        ----
        Support loading from directory containing images

        Args
        ----
        path (str) : path to the directory containing train and test pickle files
        """
        self.path = path
        self.Xtrain = pipe_utils.load_pickled_data(path + "xtrain.p")
        self.Xtest = pipe_utils.load_pickled_data(path + "xtest.p")
        self.Ytrain = pipe_utils.load_pickled_data(path + "ytrain.p")
        self.Ytest = pipe_utils.load_pickled_data(path + "ytest.p")


    def get_train_data(self):
        """
        Returns 
        -------
        X training data, Y training data """
        return self.Xtrain, self.Ytrain


    def get_test_data(self):
        """
        Returns 
        -------
        X testing data, Y testing data
        """
        return self.Xtest, self.Ytest


    def decode_class_id(self, class_ids):
        """Get the verbose name 

        Args
        ----
        class_ids (list : int) : a list of the integer representations of a class

        Returns
        -------
        (list : str) : the verbose name of each label
        """

        class_ids = pipe_utils.atleast_list(class_ids)

        for class_id in class_ids:
            for _class, _id in self.class_to_id.items():
                if _id == class_id:
                    return _class


