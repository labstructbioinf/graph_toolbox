import os
import pytest
from biopandas.pdb import PandasPdb

from graph_toolbox.feature.calc import read_struct
from graph_toolbox.feature import GraphData

test_dir = "tests/data"


def get_test_pdbs() -> list[str]:
    """get all pdb test files"""
    return [os.path.join(test_dir, fn) for fn in os.listdir(test_dir)]


testfile = "tmp.p"


@pytest.fixture(autouse=True)
def remove_cache():
    if os.path.isfile(testfile):
        os.remove(testfile)


@pytest.mark.parametrize("pdb", get_test_pdbs())
@pytest.mark.parametrize("t", [5, 7, 9])
def test_read_struct(pdb, t):

    pdb_content = PandasPdb().read_pdb(pdb).df["ATOM"]
    d = read_struct(pdb_content, t=t, with_interactions=False)
    djson = d.asdict()
    assert djson["u"].shape == djson["v"].shape
    d_interactions = read_struct(pdb_content, t=t, with_interactions=True)
    # number of edges is the same
    assert d_interactions.efeats.shape[0] == d.efeats.shape[0]
    # but number of features not
    assert d_interactions.efeats.shape[1] > d.efeats.shape[1]


@pytest.mark.parametrize("pdb", get_test_pdbs())
def test_graphobject(pdb):

    pdb_content = PandasPdb().read_pdb(pdb).df["ATOM"]
    data = GraphData.from_pdb(path=pdb_content, code=f"code_{pdb}")
    # test conversion to dgl
    data.save(testfile)
    graph = data.to_dgl()
    data_new = GraphData.load(testfile)
    graph_new = data_new.to_dgl()


# @pytest.mark.parametrize("pdb", ["tests/data/3sxw.pdb.gz", "tests/data/6iii.pdb"])
# def test_graphobject_methods(pdb):

#     data = GraphData.from_pdb(pdb)
#     graph = data.to_dgl()
#     edgedf = data.to_edgedf()
#     nodedf = data.to_nodedf()
