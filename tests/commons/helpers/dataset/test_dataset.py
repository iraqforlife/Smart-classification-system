from unittest import TestCase

import cv2
import numpy as np
import os

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.galaxy_dataset.image_strategy import GalaxyDataSetImageStrategy


class TestDataSet(TestCase):

    def setUp(self):
        # Test batch size.
        self.batch_size = 64

        # Test CSV path.
        self.path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy.csv"

        # Load data set.
        galaxy_data_set_strategy = GalaxyDataSetImageStrategy()
        self.context = Context(galaxy_data_set_strategy)
        self.dataset = self.context.load_dataset(csv_file=self.path, one_hot=True, validation_size=np.float32(0.2))

    def test_load_images(self):
        # Get a batch of data.
        x_batch, y_true_batch = self.dataset.train.next_image_batch(self.batch_size)

        # Load images in this batch.
        self.dataset.train.load_images(x_batch)

        # Get the first image ID in the training data set.
        test_dataset_img_name = self.dataset.train._img_names[0][0]

        # Get the reference image corresponding to the first image of data set.
        reference_dataset_img_name = self.dataset.train._images[0]

        # Get the path of the first image ID.
        path = os.environ["VIRTUAL_ENV"] + "/data/images/" + str(test_dataset_img_name) + ".jpg"

        # Load image.
        test_dataset_image = cv2.imread(path)

        # Transform image as a 32-bit numpy array.
        test_dataset_image = test_dataset_image.astype(np.float32)

        # Normalize the image.
        test_dataset_image = np.multiply(test_dataset_image, 1.0 / 255.0)

        np.testing.assert_array_equal(reference_dataset_img_name, test_dataset_image)

    def test_next_batch(self):
        # First batch
        x_batch1, y_true_batch1 = self.dataset.train.next_image_batch(self.batch_size)

        # Second batch
        x_batch2, y_true_batch2 = self.dataset.train.next_image_batch(self.batch_size)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_batch1, x_batch2)

    def test_validation_size(self):
        dataset = self.context.load_dataset(csv_file=self.path, one_hot=True, validation_size=np.float32(0))

        self.assertTrue(dataset.train._num_examples == 31364)

        self.assertRaises(Exception, lambda:
                          self.context.load_dataset(csv_file=self.path, one_hot=True, validation_size=np.float32(-0.1)))
