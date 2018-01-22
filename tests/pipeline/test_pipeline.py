import numpy as np
from numpy import testing as nptest
import unittest

from pipeline import utils
from pipeline.raw_data import clean_uc_merced


class UtilsTestCase(unittest.TestCase):


    def test_class_to_id(self):
        """Assert that an id is correctly mapped to it's given class label 
        """
        labels = list(utils.CLASS_TO_ID.keys())
        true_labels = np.random.choice(labels, 15)
        true_ids = [utils.CLASS_TO_ID[label] for label in true_labels]
        converted_labels = utils.ids_to_classes(true_ids)
        nptest.assert_array_equal(true_labels, converted_labels)


    @unittest.skip("Unfinished")
    def test_unnormalized_image(self):
        """Make sure that an unnormalized image gets normalized correctly 
        """
        assert False


    def test_class_ids_are_unique(self):
        """Make sure that two classes didn't get assigned the same id
        """
        labels = list(set(utils.CLASS_TO_ID.values()))
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
        pipeline_classes = utils.CLASS_TO_ID.keys()
        self.assertCountEqual(pipeline_classes, export_classes)


    @unittest.skip("Unfinished")
    def test_multiple_merced_labels_to_pipeline(self):
        assert False


    @unittest.skip("Unfinished")
    def test_correct_number_of_labeld_generated_for_test_train_sets(self):
        assert False


    @unittest.skip("Unfinished")
    def test_none_not_converted_to_label(sefl):
        """make sure that the none value is not converted to a label 
        in convert_classes
        """
        assert False



if __name__ == '__main__':
    unittest.main()
