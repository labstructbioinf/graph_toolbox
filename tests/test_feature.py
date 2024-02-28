import os
import pytest

from feature import read_struct
from feature import GraphData


testfile = "tmp.p"

@pytest.fixture(autouse=True)
def remove_cache():
    if os.path.isfile(testfile):
        os.remove(testfile)

@pytest.mark.parametrize("pdb", [
    "tests/data/3sxw.pdb.gz",
    "tests/data/6iii.pdb",
    "/home/db/localpdb/biounit/oy/2oy8.pdb.gz",
    "/home/db/localpdb/biounit/a8/1a8l.pdb.gz",
    "/home/db/localpdb/biounit/08/108l.pdb.gz"
])
def test_calc(pdb):

    u, v, feats, nfeats, struct_sequence = read_struct(pdb)


@pytest.mark.parametrize("pdb", [
    "tests/data/3sxw.pdb.gz",
    "tests/data/6iii.pdb"
])
def test_graphobject(pdb):

    data = GraphData.from_pdb(pdb)
    graph = data.to_dgl()
    data.save("tmp.pdb")
    data_new = GraphData.load("tmp.pdb")
    graph_new = data_new.to_dgl()


@pytest.mark.parametrize("pdb", [
    "tests/data/3sxw.pdb.gz",
    "tests/data/6iii.pdb"
])
def test_graphobject_methods(pdb):

    data = GraphData.from_pdb(pdb)
    graph = data.to_dgl()
    edgedf = data.to_edgedf()
    nodedf = data.to_nodedf()

