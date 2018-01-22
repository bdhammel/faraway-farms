import numpy as np
from pipeline import utils


class DataSet:

    class_to_id = utils.CLASS_TO_ID

    def __init__(self, path="./"):
        self.path = path
        self._load_train_and_test_data(path)

    def _load_train_and_test_data(self, path):
        """
        INCOMPLETE
        """
        self.Xtrain = utils.load_pickled_data(path + "xtrain.p")
        self.Xtest = utils.load_pickled_data(path + "xtest.p")
        self.Ytrain = utils.load_pickled_data(path + "ytrain.p")
        self.Ytest = utils.load_pickled_data(path + "ytest.p")


    def get_train_data(self):
        """
        """
        return self.Xtrain, self.Ytrain


    def get_test_data(self):
        """
        """
        return self.Xtest, self.Ytest


    def decode_class_id(self, class_ids):
        """Get the verbose name 
        """

        class_ids = np.atleast_1d(class_ids)

        for class_id in class_ids:
            for _class, _id in self.class_to_id.items():
                if _id == class_id:
                    return _class



if __name__ == "__main__":

    ds = DataSet(path="../../datasets/uc_merced/")


