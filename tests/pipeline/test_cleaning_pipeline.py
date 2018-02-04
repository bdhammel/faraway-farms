import numpy as np
from numpy import testing as nptest
import unittest

from pipeline import utils as pipe_utils
from pipeline.raw_data import clean_uc_merced
from pipeline.raw_data import clean_dstl


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
