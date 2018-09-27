from unittest import TestCase

import numpy as np
import os

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.spam_dataset.feature_strategy import SpamDataSetFeatureStrategy


class TestSpamDataSetFeatureStrategy(TestCase):
    def setUp(self):
        self.path = os.environ["VIRTUAL_ENV"] + "/data/csv/spam/spam.csv"

    def test_load_dataset_no_oneHot(self):
        galaxy_data_set_strategy = SpamDataSetFeatureStrategy()
        context = Context(galaxy_data_set_strategy)
        dataset = context.load_dataset(csv_file=self.path, one_hot=False, validation_size=np.float32(0.2))
        self.assertTrue(dataset.train._num_examples == 2208)


    def test_load_dataset_with_oneHot(self):
        galaxy_data_set_strategy = SpamDataSetFeatureStrategy()
        context = Context(galaxy_data_set_strategy)
        dataset = context.load_dataset(csv_file=self.path, one_hot=True, validation_size=np.float32(0.2))
        self.assertTrue(dataset.train._num_examples == 2208)

