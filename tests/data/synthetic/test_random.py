# pylint: disable=W0212
from datetime import datetime as dt

import pytest
import numpy as np

import mltools.data.synthetic.random as random


@pytest.mark.parametrize("length", [(3), (6), (12)])
def test_rand_alphanum_item(length):
    r = random.rand_alphanum_item(length=length)

    assert len(r) == length


def test_rand_alphanum_item_0_len():
    with pytest.raises(ValueError):
        random.rand_alphanum_item(length=0)


@pytest.mark.parametrize(
    "start, end, fmt",
    [
        ("01-01-2000", "12-31-2000", "%m-%d-%Y"),
        ("0100", "1200", "%m%y"),
        ("01-01-00 11:00:00", "01-01-00 12:00:00", "%m-%d-%y %H:%M:%S")
    ]
)
def test_rand_date_item(start, end, fmt):
    r = random.rand_date_item(start=start, end=end, fmt=fmt)
    assert dt.strptime(start, fmt) <= dt.strptime(r, fmt) <= dt.strptime(end, fmt)


@pytest.mark.parametrize("val, expected", [((0, 1, 2), 0), ([1, 2, 3], 6)])
def test_iterable_multiply(val, expected):
    assert random._iterable_multiply(val) == expected


@pytest.mark.parametrize(
    "size, expected",
    [
        (None, 0),
        (5, np.array([0,0,0,0,0])),
        ((2, 3), np.array([0, 0, 0, 0, 0, 0]).reshape(2, 3))
    ]
)
def test_ndim_random(size, expected):
    def zeros():
        return 0

    cond = random._ndim_random(zeros, size=size) == expected
    if size is None:
        assert cond
    else:
        assert cond.all()
