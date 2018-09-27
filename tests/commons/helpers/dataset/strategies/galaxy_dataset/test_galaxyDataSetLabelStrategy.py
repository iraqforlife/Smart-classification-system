from unittest import TestCase

import os
import numpy as np

from commons.helpers.dataset.context import Context
from commons.helpers.dataset.strategies.galaxy_dataset.label_strategy import GalaxyDataSetLabelStrategy


class TestGalaxyDataSetLabelStrategy(TestCase):

    def setUp(self):
        self.path = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy.csv"

    def test_load_dataset_no_oneHot(self):
        galaxy_data_set_strategy = GalaxyDataSetLabelStrategy()
        context = Context(galaxy_data_set_strategy)
        dataset = context.load_dataset(csv_file=self.path, one_hot=False, validation_size=np.float32(0.2))
        self.assertTrue(dataset.train._num_examples == 25091)

    def test_load_dataset_with_oneHot(self):
        galaxy_data_set_strategy = GalaxyDataSetLabelStrategy()
        context = Context(galaxy_data_set_strategy)
        dataset = context.load_dataset(csv_file=self.path, one_hot=True, validation_size=np.float32(0.2))
        self.assertTrue(dataset.train._num_examples == 25091)
