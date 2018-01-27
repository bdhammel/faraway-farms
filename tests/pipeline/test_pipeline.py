import numpy as np
from numpy import testing as nptest
import unittest

from pipeline import utils as pipe_utils
from pipeline.raw_data import clean_uc_merced
from pipeline.raw_data import clean_dstl


class UtilsTestCase(unittest.TestCase):


    def test_class_to_id(self):
        """Assert that an id is correctly mapped to it's given class label 
        """
        labels = list(pipe_utils.CLASS_TO_ID.keys())
        true_labels = np.random.choice(labels, 15)
        true_ids = [pipe_utils.CLASS_TO_ID[label] for label in true_labels]
        converted_labels = pipe_utils.ids_to_classes(true_ids)
        nptest.assert_array_equal(true_labels, converted_labels)


    @unittest.skip("Unfinished")
    def test_unnormalized_image(self):
        """Make sure that an unnormalized image gets normalized correctly 

        - check channel last, 16 bit, and normalization [0,1)
        """
        img = pipe_utils.imread("../extras/16bit.tif")

        # sanity check
        self.assertGreater(img.max(), 255)

        # Assert normalized
        clean_img = pipe_utils.image_preprocessor(img)
        self.assertLess(img.max(), 1.)
        self.assertGreater(img.min(), 0.)

        # assert channel last
        self.assertEqual(img.shape[-1], 3)


    def test_class_ids_are_unique(self):
        """Make sure that two classes didn't get assigned the same id
        """
        labels = list(set(pipe_utils.CLASS_TO_ID.values()))
        assert len(labels) == max(labels)+1


    def test_undesired_labels_arnt_loaded(self):
        assert False


class UCMercedTestCase(unittest.TestCase):


    def test_export_classes_match_pipeline_classes(self):
        """
        """

        # Don't brink in the None values, check that these aren't imported in 
        # a separate test
        export_classes = [value for value in clean_uc_merced.MAP_TO_LOCAL_LABELS.values() if value]
        pipeline_classes = pipe_utils.CLASS_TO_ID.keys()
        self.assertCountEqual(pipeline_classes, export_classes)


    @unittest.skip("Unfinished")
    def test_multiple_merced_labels_to_pipeline(self):
        assert False


    @unittest.skip("Unfinished")
    def test_correct_number_of_labeld_generated_for_test_train_sets(self):
        assert False


    @unittest.skip("Unfinished")
    def test_none_not_converted_to_label(self):
        """make sure that the none value is not converted to a label 
        in convert_classes
        """
        assert False


class DSTLTestCase(unittest.TestCase):


    def setUp(cls):
        cls.img_path = "../extras/dstl_etc/images/16bit.tif"
        cls.loader = clean_dstl.dstl_loader(
                geojson_dir=geojson_dir, 
                grid_sizes=grid_sizes
        )
        grid_file = "../extras/dstl_etc/grid_sizes.csv"
        geojson_dir = "../extras/dstl_etc/geojson"
        grid_sizes = import_grid_sizes(grid_file)


    def test_image_id_extracted_correctly(self):
        """Make sure the image id is correctly extracted from the path

        image id is just the file name, no extension 
        """
        img = clean_dstl.DSTLImage(self.path)
        self.assertEqual(img.image_id, '16bit')


    def test_no_geo_json_for_img(self):
        img = self.loader()

        with self.assertRaises(Exception):
            pass


    def test_dstl_labels_maped_to_local_labels(self):
        assert False


if __name__ == '__main__':
    unittest.main()
