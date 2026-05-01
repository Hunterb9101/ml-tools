# pylint: disable=W0212
from datetime import UTC
from datetime import datetime as dt

import numpy as np
import pytest

from mltools.data.synthetic import random


@pytest.mark.parametrize("length", [(3), (6), (12)])
def test_rand_alphanum_item(length):
    r = random.rand_alphanum_item(length=length)

    assert len(r) == length


def test_rand_alphanum_item_0_len():
    with pytest.raises(ValueError, match="Invalid length"):
        random.rand_alphanum_item(length=0)


@pytest.mark.parametrize(
    ("start", "end", "fmt"),
    [
        ("01-01-2000", "12-31-2000", "%m-%d-%Y"),
        ("0100", "1200", "%m%y"),
        ("01-01-00 11:00:00", "01-01-00 12:00:00", "%m-%d-%y %H:%M:%S"),
    ],
)
def test_rand_date_item(start, end, fmt):
    r = random.rand_date_item(start=start, end=end, fmt=fmt)
    start_dt = dt.strptime(start, fmt).replace(tzinfo=UTC)
    result_dt = dt.strptime(r, fmt).replace(tzinfo=UTC)
    end_dt = dt.strptime(end, fmt).replace(tzinfo=UTC)
    assert start_dt <= result_dt <= end_dt


@pytest.mark.parametrize(("val", "expected"), [((0, 1, 2), 0), ([1, 2, 3], 6)])
def test_iterable_multiply(val, expected):
    assert random._iterable_multiply(val) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (None, 0),
        (5, np.array([0, 0, 0, 0, 0])),
        ((2, 3), np.array([0, 0, 0, 0, 0, 0]).reshape(2, 3)),
    ],
)
def test_ndim_random(size, expected):
    def zeros():
        return 0

    cond = random._ndim_random(zeros, size=size) == expected
    if size is None:
        assert cond
    else:
        assert cond.all()
