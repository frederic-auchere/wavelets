import pytest
import numpy as np
from watroo import AtrousTransform


@pytest.fixture
def image_2d():
    image_2d = np.ones((128, 128))
    return image_2d


class TestTransform:
    def test_regular_recursive(self, image_2d):
        transform = AtrousTransform()
        regular = transform(image_2d, 4)
        recursive = transform(image_2d, 4, recursive=True)
        assert np.isclose(regular, recursive).all()
