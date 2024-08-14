"""Unit tests for generators/"""

# Author: Kyle Nakamura
# License: BSD 3 clause

import numpy as np
import unittest
from mlrose_hiive.samples.synthetic_data import SyntheticDataGenerator, plot_synthetic_dataset
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

try:
    import mlrose_hiive
except ImportError:
    import sys

    sys.path.append("..")
    import mlrose_hiive
class TestSyntheticDataGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = SyntheticDataGenerator(seed=42)

    def test_get_synthetic_features_and_classes(self):
        features, classes = self.generator.get_synthetic_features_and_classes()
        self.assertEqual(features, ["(1) A", "(2) B"])
        self.assertEqual(classes, ["RED", "BLUE"])

        features, classes = self.generator.get_synthetic_features_and_classes(with_redundant_column=True)
        self.assertEqual(features, ["(1) A", "(2) B", "(3) R"])
        self.assertEqual(classes, ["RED", "BLUE"])

    def test_get_synthetic_data(self):
        data, features, classes, output_directory = self.generator.get_synthetic_data()
        self.assertEqual(data.shape, (400, 3))
        self.assertEqual(features, ["(1) A", "(2) B"])
        self.assertEqual(classes, ["RED", "BLUE"])
        self.assertIsNone(output_directory)

    def test_setup_synthetic_data_test_train(self):
        data, _, _, _ = self.generator.get_synthetic_data()
        x, y, x_train, x_test, y_train, y_test = self.generator.setup_synthetic_data_test_train(data)
        self.assertEqual(x_train.shape, (280, 2))
        self.assertEqual(x_test.shape, (120, 2))
        self.assertEqual(y_train.shape, (280,))
        self.assertEqual(y_test.shape, (120,))

    def test_plot_synthetic_dataset(self):
        data, _, _, _ = self.generator.get_synthetic_data()
        x, y, x_train, x_test, y_train, y_test = self.generator.setup_synthetic_data_test_train(data)
        classifier = LogisticRegression().fit(x_train, y_train)
        plt.figure()
        plot_synthetic_dataset(x_train, x_test, y_train, y_test, classifier=classifier)
        plt.close()

if __name__ == "__main__":
    unittest.main()
