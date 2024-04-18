import pytest
import numpy as np


@pytest.fixture
def data_2d():
    return np.ones((128, 128))
