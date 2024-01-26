import pytest

from feature import read_struct
from feature import GraphData


@pytest.mark.parametrize("pdb", [
    "tests/data/3sxw.pdb.gz",
    "tests/data/6iii.pdb",
    "tests/data/1b5s.pdb.gz"
])
def test_calc(pdb):

    u, v, feats, r = read_struct(pdb, chain="E")

@pytest.mark.parametrize("pdb", [
    "tests/data/3sxw.pdb.gz",
    "tests/data/6iii.pdb"
])
def test_graphobject(pdb):

    data = GraphData.from_pdb(pdb)
