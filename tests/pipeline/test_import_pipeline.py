"""Testing the importing of data into the model

This test file tests attributes of the pipeline used to load cleaned images, 
pre-process them, and behavior before they're fed into the model.

This includes tests on both object identification and image classification

i.e. 
pipeline.patch_pipeline
pipeline.obj_pipeline
pipeline.utils

"""

import numpy as np
from numpy import testing as nptest
import unittest

from pipeline import utils as pipe_utils
from pipeline import obj_pipeline
from pipeline import patch_pipeline


class SatalliteImageTestCase(unittest.TestCase):
    
    def test_data_not_loaded_raises_exception(self):
        pass

    def assert_data_is_ok_is_called(self):
        pass


class ImagePreprocessTestCase(unittest.TestCase):
    pass


class DataCheckTestCase(unittest.TestCase):

    def test_no_data_id_raises_exception(self):
        pass

    def test_wrongly_shaped_image_raises_exception(self):
        pass

    def test_image_is_the_right_dtype(self):
        pass

    def test_image_is_in_wrong_range(self):


class GenerateTrainTestSetTestCase(unittest.TestCase):

    def assert_data_is_ok_is_called(self):
        pass


class UtilsTestCase(unittests.TestCase):

    def test_item_converted_to_list(self):
        pass

    def test_path_name_correctly_extracted(self):
        pass



class ObjTestCase(unittest.TestCase):

    def test_data_loaded_does_not_raise_exception_in_parent(self):
        pass

    def test_feature_can_be_appended(self):
        pass

    def test_instance_has_correct_labels(self):
        pass


class ObjPipelineTestCase(unittest.TestCase):
    pass


class PatchTestCase(unittest.TestCase):
    pass
