import pytest

from minimeds_sage import TensorSpace


@pytest.fixture
def space():
    return TensorSpace(q=101, n=6)
