import pytest

from feature import read_struct


@pytest.mark.parametrize("pdb", [
    "tests/data/3sxw.pdb.gz",
    "tests/data/6iii.pdb"
])
def test_calc(pdb):

    u, v, feats, r = read_struct(pdb)
