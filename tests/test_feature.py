import pytest

from feature import read_struct
from feature import GraphData


@pytest.mark.parametrize("pdb", [
    "tests/data/3sxw.pdb.gz",
    "tests/data/6iii.pdb",
    "/home/db/localpdb/biounit/oy/2oy8.pdb.gz",
    "/home/db/localpdb/biounit/a8/1a8l.pdb.gz",
    "/home/db/localpdb/biounit/08/108l.pdb.gz"
])
def test_calc(pdb):

    u, v, feats, r = read_struct(pdb)

@pytest.mark.parametrize("pdb", [
    "tests/data/3sxw.pdb.gz",
    "tests/data/6iii.pdb"
])
def test_graphobject(pdb):

    data = GraphData.from_pdb(pdb)
