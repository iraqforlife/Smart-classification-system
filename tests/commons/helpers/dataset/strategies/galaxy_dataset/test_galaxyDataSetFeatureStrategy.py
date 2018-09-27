from unittest import TestCase

import numpy as np
import os

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.galaxy_dataset.feature_strategy import GalaxyDataSetFeatureStrategy


class TestGalaxyDataSetFeatureStrategy(TestCase):
    def setUp(self):
        self.path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy_feature_vectors.csv"
        self.validation_size = np.float32(0.2)

    def test_load_dataset_no_oneHot(self):
        galaxy_data_set_strategy = GalaxyDataSetFeatureStrategy()
        context = Context(galaxy_data_set_strategy)
        dataset = context.load_dataset(csv_file=self.path, one_hot=False, validation_size=self.validation_size)
        self.assertTrue(dataset.train._num_examples == 25091)

    def test_load_dataset_with_oneHot(self):
        galaxy_data_set_strategy = GalaxyDataSetFeatureStrategy()
        context = Context(galaxy_data_set_strategy)
        dataset = context.load_dataset(csv_file=self.path, one_hot=True, validation_size=self.validation_size)
        self.assertTrue(dataset.train._num_examples == 25091)
