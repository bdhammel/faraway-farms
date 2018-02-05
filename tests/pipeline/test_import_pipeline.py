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
from unittest.mock import MagicMock, patch

from pipeline import utils as pipe_utils
from pipeline import obj_pipeline
from pipeline import patch_pipeline


class SatalliteImageTestCase(unittest.TestCase):

    def setUp(cls):
        # Generate fake data
        cls.faux_data = np.zeros(shape=(200,200,3))
    

    def test_no_label_loaded_raises_exception(self):
        """Make sure an image has an id
        The image id becomes the file name if the image is saved, make sure an 
        exception is raise if one doesn't exist
        """
        im = pipe_utils.SatelliteImage(self.faux_data)

        with self.assertRaises(Exception):
            image_id = im.image_id


    @patch('pipeline.utils.data_is_ok')
    def test_data_is_ok_is_called(self, mock):
        """Make sure the data check function is called, to ensure that dirty 
        data isn't loaded into a SatalliteImage class. 

        It should be assumed that any data in this class is clean and ready to 
        go into a model (pre-model processing) i.e. data should be [0,255]
        and the correct size
        """
        im = pipe_utils.SatelliteImage(self.faux_data)
        self.assertTrue(mock.called)


class ImagePreprocessTestCase(unittest.TestCase):


    def test_alpha_channel_dropped(self):
        """test that a 4th color channel, the alpha channel, is dropped from the
        image
        """
        faux_data = 100*np.ones(shape=(300,300,4), dtype=np.uint8)
        # sanity check
        self.assertTrue(faux_data.shape[-1] == 4)

        cleaned_data = pipe_utils.image_save_preprocessor(faux_data, report=False)
        self.assertTrue(cleaned_data.shape[-1] == 3)

        # make sure values weren't changed
        nptest.assert_array_equal(cleaned_data, faux_data[...,:3])


    def test_bw_to_color(self):
        """Check that a BW image is turned to RGB color channels
        """
        faux_data = 100*np.ones(shape=(300,300,1), dtype=np.uint8)
        # sanity check
        self.assertTrue(faux_data.shape[-1] == 1)

        cleaned_data = pipe_utils.image_save_preprocessor(faux_data, report=False)
        self.assertTrue(cleaned_data.shape[-1] == 3)

        # test, 2D BW image also works 

        faux_data = 100*np.ones(shape=(300,300), dtype=np.uint8)
        # sanity check
        self.assertTrue(faux_data.ndim == 2)

        cleaned_data = pipe_utils.image_save_preprocessor(faux_data, report=False)
        self.assertTrue(cleaned_data.shape[-1] == 3)


    def test_16_bit(self):
        """Check that a 16 bit image is downsampled to 8 bit
        """
        faux_data = 65000*np.ones(shape=(300,300,3), dtype=np.uint16)

        cleaned_data = pipe_utils.image_save_preprocessor(faux_data, report=False)
        self.assertTrue(cleaned_data.max() <= 255)


    def test_8_bit(self):
        pass


    def test_0_to_1_normalized(self):
        """Check that if an image range is [0,1) it is re sampled to be [0,255]
        """
        faux_data = np.ones(shape=(300,300,3), dtype=np.uint16)

        cleaned_data = pipe_utils.image_save_preprocessor(faux_data, report=False)
        self.assertTrue(cleaned_data.max() <= 255)
        self.assertTrue(cleaned_data.max() > 1)
    

    def test_channel_first(self):
        """Test that if color channel is first, the image is reorder to color
        channel last
        """
        faux_data = 100*np.ones(shape=(3,300,300), dtype=np.uint8)

        cleaned_data = pipe_utils.image_save_preprocessor(faux_data, report=False)

        nptest.assert_array_equal(cleaned_data.shape, (300,300,3))


class DataCheckTestCase(unittest.TestCase):


    def test_wrongly_shaped_image_raises_exception(self):
        """Make sure an exception is raised if the data in not of the correct
        shape
        """
        faux_data = 100*np.ones(shape=(300,300,3), dtype=np.uint8)

        with self.assertRaises(Exception):
            pipe_utils.data_is_ok(faux_data, use='patch', raise_exception=True)


    def test_data_check_doesnt_have_to_raise_exception(self):
        """Repeat test_wrongly_shaped_image_raises_exception just to make sure
        that an exception doesnt have to be raise, and that a 'False' is 
        returned instead
        """
        faux_data = 100*np.ones(shape=(300,300,3), dtype=np.uint8)

        self.assertFalse(pipe_utils.data_is_ok(faux_data, use='patch'))


    def test_obj_data_shape_is_ok(self):
        """Make sure data check returns true for an obj data shape
        """
        faux_data = 100*np.ones(shape=(400,400,3), dtype=np.uint8)

        self.assertTrue(pipe_utils.data_is_ok(faux_data, use='obj'))


    def test_patch_data_shape_is_ok(self):
        """Make sure data check returns true for an obj data shape
        """
        faux_data = 100*np.ones(shape=(200,200,3), dtype=np.uint8)

        self.assertTrue(pipe_utils.data_is_ok(faux_data, use='patch'))


    def test_image_is_the_right_dtype(self):
        """Make sure the images are of type uint8 i.e. [0,255] range
        """
        faux_data = 100*np.ones(shape=(200,200,3), dtype=np.float32)

        with self.assertRaises(Exception):
            pipe_utils.data_is_ok(faux_data, use='patch', raise_exception=True)


    def test_image_is_in_wrong_range(self):
        """See that if a image mistakenly normalized to [0,1), and then converted
        to uint8 raises an alert
        """
        faux_data = np.ones(shape=(200,200,3), dtype=np.uint8)

        with self.assertRaises(Exception):
            pipe_utils.data_is_ok(faux_data, use='patch', raise_exception=True)


    def test_image_has_alpha_channel(self):
        """Make sure that an image with an alpha channel doesn't pass the data
        check
        """
        faux_data = np.ones(shape=(200,200,4), dtype=np.uint8)

        with self.assertRaises(Exception):
            pipe_utils.data_is_ok(faux_data, use='patch', raise_exception=True)


class GenerateTrainTestSetTestCase(unittest.TestCase):


    @patch('pipeline.utils.data_is_ok')
    def assert_data_is_ok_is_called(self):
        pass


class UtilsTestCase(unittest.TestCase):
    """General tests for functions within pipeline.utils
    """

    def test_single_item_converted_to_list(self):
        """check that a single item passes to 'atleast_list' is converted to 
        a list
        """
        true_item = 'test'
        islist = pipe_utils.atleast_list(true_item)

        self.assertTrue(isinstance(islist, list))


    def test_list_passed_to_as_list_does_noting(self):
        """Make sure atleat_list doesn't make a list of lists
        """
        true_item = ['test']
        islist = pipe_utils.atleast_list(true_item)
        self.assertTrue(true_item[0], islist[0])


    def test_path_name_correctly_extracted(self):
        """Check that only the file name, no extension is returned
        from  get_file_name_from_path
        """
        path = 'this/is/a/fake/path/name.ext'
        name = pipe_utils.get_file_name_from_path(path)
        self.assertEqual(name, 'name')


    @unittest.skip("Capture Stdout")
    @patch('pipeline.utils.data_is_ok')
    def test_pickle_dump_calls_data_check(self):
        pass


    @unittest.skip("Mock pickle.load")
    @patch('pipeline.utils.data_is_ok')
    def test_pickle_load_calls_data_check(self):
        pass


    def test_ids_to_classes(self):
        """test the correct verbose labels are returned given an id input
        Important for correctly identifying a prediction at the output of a
        model
        """
        # Generate random selection of ids, and convert back to labels 
        class_to_id = pipe_utils.PATCH_CLASS_TO_ID
        true_labels = list(np.random.choice(list(class_to_id.keys()), 10))
        ids = [class_to_id[label] for label in true_labels]
        check_labels = pipe_utils.ids_to_classes(ids, use='patch')

        self.assertListEqual(true_labels, check_labels)



class ObjTestCase(unittest.TestCase):

    def setUp(cls):
        # Generate fake data
        cls.faux_data = np.zeros(shape=(200,200,3))
        cls.faux_feature = {'name':'label', 'coors':[(1,2,3,4), (10,20,30,40)]}


    def test_data_is_correctly_loaded(self):
        """Sanity check to make sure data is loaded into the parent class
        Satellite image
        """
        img = obj_pipeline.ObjImage(data=self.faux_data)
        nptest.assert_array_equal(self.faux_data, img.data)


    @unittest.skip("Mock path loading")
    def test_data_loaded_from_path_correctly(self):
        pass


    def test_feature_can_be_appended(self):
        """Make sure a feature can be appended correctly

        TODO: Also test multiple appending 
        """
        img = obj_pipeline.ObjImage(data=self.faux_data)
        img.append_feature(self.faux_feature['name'], self.faux_feature['coors'])
        returned_feature = img.get_features()
        self.assertTrue(self.faux_feature['name'] in list(returned_feature.keys()))


    def test_instance_has_correct_labels(self):
        pass


    @patch('pipeline.utils.data_is_ok')
    def test_data_is_ok_is_called(self, mock):
        """There's no reason that this function shouldn't be called, because 
        its contained within the parent class SatelliteImage, but this is a sanity
        check
        """
        img = obj_pipeline.ObjImage(data=self.faux_data)
        self.assertTrue(mock.called)


class ObjPipelineTestCase(unittest.TestCase):
    

    @unittest.skip("Figure out how to mock the csv file")
    def test_updateing_annotation_file_img_paths(self):
        """check that the file names within an annotation path can be updated 
        to a new directory
        """
        pass


    @unittest.skip("Figure out how to mock the csv file")
    def test_two_annotations_files_can_be_merged(self):
        """See that two files can be combined correctly
        """
        pass


class PatchTestCase(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()


