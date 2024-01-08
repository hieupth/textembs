import pytest
import textembserve


def test_sum_as_string():
    assert textembserve.sum_as_string(1, 1) == "2"
