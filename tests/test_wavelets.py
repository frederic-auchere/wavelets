from . import data_2d
import numpy as np
from watroo import AtrousTransform


class TestTransform:

    def test_regular(self, data_2d):
        transform = AtrousTransform()
        regular = transform(data_2d, 4)
        expected = np.zeros(regular.data.shape)  # first planes should be zeros
        expected[-1] = 1  # last planes should be ones
        assert np.isclose(regular, expected).all()

    def test_regular_vs_recursive(self, data_2d):
        transform = AtrousTransform()
        regular = transform(data_2d, 4)
        recursive = transform(data_2d, 4, recursive=True)
        assert np.isclose(regular, recursive).all()
