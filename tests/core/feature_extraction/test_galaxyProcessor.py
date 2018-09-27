#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from unittest import TestCase
from core.feature_extraction.galaxy.galaxy_processor import GalaxyProcessor
from commons.helpers.dataset.strategies.galaxy_dataset.label_strategy import GalaxyDataSetLabelStrategy
from commons.helpers.dataset.context import Context

class TestGalaxyProcessor(TestCase):
    def setUp(self):

        validation_size = 0.2
        # Get the ground truth CSV file from script's parameters.
        self.galaxy_csv_file = os.environ["VIRTUAL_ENV"] + "/data/csv/galaxy/galaxy.csv"
        self.galaxy_images_path = os.environ["VIRTUAL_ENV"] + "/data/images/"

        # Create instance of data set loading strategies.
        galaxy_label_data_set_strategy = GalaxyDataSetLabelStrategy()

        # Set the context to galaxy label data set loading strategy.
        context = Context(galaxy_label_data_set_strategy)
        context.set_strategy(galaxy_label_data_set_strategy)
        self.label_dataset = context.load_dataset(csv_file=self.galaxy_csv_file, one_hot=False,
                                             validation_size=np.float32(validation_size))

    def testGalaxyProcessor(self):

        # Process galaxies.
        galaxy_processor = GalaxyProcessor(self.galaxy_images_path)
        #features = galaxy_processor.process_galaxy(self.label_dataset)