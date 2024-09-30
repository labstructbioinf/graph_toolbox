# %%
import os
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from tqdm import tqdm
import  matplotlib.pyplot as plt

from feature.base import GraphData
version = "v2"
basedir = f"/home/nfs/kkaminski/deepsocket/data/{version}"
output = "pdbdata"

path_h5 = f"{basedir}/pdb/pdb.h5"
path_index = f"{basedir}/pdb/pdb.clean.p.gz"

assert os.path.join(path_h5)
assert os.path.join(path_index)
os.makedirs(output, exist_ok=True)
# %%
data = pd.read_pickle(path_index)
data = data[data.is_valid == True].copy()
data = data.sample(200)
# %%
# %%
datatmp = list()
for idx, row in tqdm(data.iterrows()):
    pdbid, chain, hnum = row.code.split("_")
    key = f"/{pdbid[:2]}/{pdbid}/{row.code}"
    atoms = pd.read_hdf(path_h5, key=key)
    try:
        gobj = GraphData.from_pdb(atoms, code=row.code)
    except:
        continue
    tmp = gobj.to_nodedf()
    tmp['key'] = key
    datatmp.append(tmp)

datat = pd.concat(datatmp, axis=0)

# %%
for key, gdata in datat.groupby(["key", "residue"]):
    print(gdata.resid.tolist())

# %%
plt.scatter(datat.psi, datat.phi)
plt.savefig('rama.png')
# %%
