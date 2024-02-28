# %%
import sys
import time

sys.path.append("..")
import pandas as pd

from feature.base import GraphData

# %%

dataset = "/home/nfs/rmadaj/DeepSocket_prepare/data/feb24/feb24_full.pkl"
dataset = pd.read_pickle(dataset)
#dataset = dataset[dataset.is_valid == True]

# %%
resdata = list()
conndata = list()
t0 = time.perf_counter()
for idx, row in dataset.iterrows():
    try:
        gr = GraphData.from_pdb(row.path)
    except Exception as e:
        print(e)
        continue
    resdata.append(gr.to_nodedf())
    conndata.append(gr.to_edgedf())
    print(idx)
    if idx > 80:
        break
time_total = time.perf_counter() - t0

print(f'parsing time: {time_total:.2f} [s] {time_total/len(resdata):.2f} itr/chain [s]')
resdf = pd.concat(resdata, ignore_index=True)
conndf = pd.concat(conndata, ignore_index=True)
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
ax = ax.ravel()
ax[0].scatter(resdf.psi, resdf.phi)
ax[0].set_xlim([-3.14, 3.14])
ax[0].set_ylim([-3.14, 3.14])
ax[0].set_xlabel('psi')
ax[0].set_ylabel('phi')
ax[0].grid()


ax[1].scatter(resdf.chi1, resdf.chi2)
ax[1].set_xlim([-3.14, 3.14])
ax[1].set_ylim([-3.14, 3.14])
ax[1].set_xlabel('chi1')
ax[1].set_ylabel('chi2')
ax[1].grid()

fig.savefig('dihedral_angle.png')
# %%

resdf.to_csv('resdata.csv')
conndf.to_csv('conndata.csv')


