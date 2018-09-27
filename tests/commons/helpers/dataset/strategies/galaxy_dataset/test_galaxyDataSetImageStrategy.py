from unittest import TestCase

import os
import numpy as np

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.galaxy_dataset.image_strategy import GalaxyDataSetImageStrategy


class TestGalaxyDataSetImageStrategy(TestCase):
    def setUp(self):
        self.path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy.csv"

    def test_load_dataset_with_oneHot(self):
        galaxy_data_set_strategy = GalaxyDataSetImageStrategy()
        context = Context(galaxy_data_set_strategy)
        dataset = context.load_dataset(csv_file=self.path, one_hot=True, validation_size=np.float32(0.2))
        self.assertTrue(dataset.train._num_examples == int(np.round(31364 * 0.8)))
        self.assertTrue(dataset.valid._num_examples == int(np.round(31364 * 0.2)))
