from . import data_2d
from watroo.utils import wow


class TestWOW:

    def test_wow(self, data_2d):
        wowed, _ = wow(data_2d)
        wowed, _ = wow(data_2d, bilateral=True)
