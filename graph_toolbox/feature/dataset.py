import os
from pathlib import Path
import time
from typing import List, Any, Union

import h5py
import numpy as np
import torch

import dgl
from .base import GraphData


def key_from_code(code: str):
    """create key from pdb_chain_hnum"""
    pdb, chain, hnum = code.split("_")
    key = f"/_{pdb[1:3]}/_{pdb}/_{code}"
    return key


def traverse_datasets(hdf_file):
    # https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
    def h5py_dataset_iterator(g, prefix=""):
        for key in g.keys():
            item = g[key]
            path = f"{prefix}/{key}"
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


class H5Handle:
    """
    for graphs
    """

    dataset_attrs = {"compression": "lzf", "dtype": np.float32}
    reqkeys = {"u", "v", "nfeats", "efeats", "sequence", "distancemx"}
    numkeys = len(reqkeys)
    error_group: str = "errors"
    direct_read: bool = True
    filename: str

    def __init__(self, filename: Union[str, Path], keep_open: bool = False):
        """
        Args:
            keep_open (bool): if True keep file open
        """
        filename = Path(filename)
        if not os.path.isfile(filename):
            with h5py.File(filename, "w") as hf:
                pass
        self.filename = filename
        self._hf = None
        if keep_open:
            self._hf = h5py.File(self.filename, "a")

    def _open(self, mode="r"):
        if self._hf is not None:
            return self._hf
        return h5py.File(self.filename, mode)

    def _close(self, hf):
        if hf is not self._hf:
            hf.close()

    def close(self):
        """Close the persistently held file handle, if any."""
        if self._hf is not None:
            self._hf.close()
            self._hf = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    # https://github.com/harvardnlp/botnet-detection/blob/master/graph_data_storage.md
    def write_graph(self, g: GraphData, direct_code: bool = False):
        """
        with use_code g.code is a direct key
        """
        hf = self._open(mode="a")
        try:
            if direct_code:
                pdbgr = hf.require_group(g.code)
            else:
                group = self.group_from_code(g.code)
                pdbgr = hf.require_group(group)
            pdbgr.attrs["is_valid"] = False
            for key, val in g.to_h5().items():
                pdbgr.create_dataset(name=key, data=val)
            pdbgr.attrs["is_valid"] = True
        finally:
            self._close(hf)

    def read_graph(self, code, with_dist=True, sdh=True):
        """
        if size is none read all record from start to the end
        """
        hf = self._open(mode="r")
        try:
            if sdh:
                key = key_from_code(code)
                pdbsubgr = hf[key]
            else:
                group = self.group_from_code(code)
                pdbsubgr = hf[group]
            try:
                u, v = torch.from_numpy(pdbsubgr["u"][:]), torch.from_numpy(
                    pdbsubgr["v"][:]
                )
                sequence = torch.from_numpy(pdbsubgr["sequence"][:])
                nfeats, efeats = torch.from_numpy(
                    pdbsubgr["nfeats"][:]
                ), torch.from_numpy(pdbsubgr["efeats"][:])
                if with_dist:
                    distancemx = torch.from_numpy(pdbsubgr["distancemx"][:])

                g = dgl.graph((u, v))
                g.ndata["seq"] = sequence
                g.ndata["angles"] = nfeats
                g.edata["f"] = efeats
            except KeyError as e:
                raise KeyError(f"missing {e} for gr {key if sdh else group}")
            if with_dist:
                return g, distancemx
            else:
                return g
        finally:
            self._close(hf)

    def read_graph_opt(self, code, with_dist=True, sdh=True):
        """
        Reads a graph from an HDF5 file and converts it into a DGLGraph.
        """
        hf = self._open(mode="r")
        try:
            key = key_from_code(code) if sdh else self.group_from_code(code)
            try:
                pdbsubgr = hf[key]
                u = torch.from_numpy(pdbsubgr["u"])
                v = torch.from_numpy(pdbsubgr["v"])
                sequence = torch.from_numpy(pdbsubgr["sequence"])
                nfeats = torch.from_numpy(pdbsubgr["nfeats"])
                efeats = torch.from_numpy(pdbsubgr["efeats"])

                g = dgl.graph((u, v), num_nodes=sequence.shape[0])
                g.ndata["seq"] = sequence
                g.ndata["angles"] = nfeats
                g.edata["f"] = efeats

                if with_dist:
                    distancemx = torch.from_numpy(pdbsubgr["distancemx"])
                    return g, distancemx
                else:
                    return g

            except KeyError as e:
                raise KeyError(f"Missing key {e} for group {key}")
        finally:
            self._close(hf)

    def read_key(self, code: str, key: str):
        hf = self._open(mode="r")
        try:
            group = self.group_from_code(code)
            return hf[group][key][:]
        finally:
            self._close(hf)

    def write_corrupted(self, code):
        hf = self._open(mode="a")
        try:
            pdbgr = hf.require_group(self.error_group)
            pdbgr.attrs["is_valid"] = False
        finally:
            self._close(hf)

    def group_from_code(self, code):
        """
        locate h5 group based on code
        """
        pdb, _, _ = code.split("_")
        preffix = pdb[:2]
        group = f"{preffix}/{pdb}/{code}"
        return group

    @property
    def pdbs(self) -> list:
        hf = self._open(mode="r")
        try:
            pdbs = list(hf.keys())
        finally:
            self._close(hf)
        return pdbs

    @property
    def codes(self) -> list:
        # https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
        hf = self._open(mode="r")
        try:
            valid_codes = list()
            preffix = hf.keys()
            for pre in preffix:
                hfpdb = hf[pre]
                pdbs = hfpdb.keys()
                for pdb in pdbs:
                    cursor = hfpdb[pdb]
                    if not isinstance(cursor, h5py.Group):
                        continue
                    codes = cursor.keys()
                    for code in codes:
                        if cursor[code].attrs.get("is_valid", False):
                            valid_codes.append(code)
        finally:
            self._close(hf)
        return valid_codes

    @property
    def invalid(self) -> list:
        hf = self._open(mode="a")
        try:
            error_grp = hf.require_group(self.error_group)
            result = list(error_grp.keys())
        finally:
            self._close(hf)
        return result


class EmbH5Handle:
    reqkeys = {"emb"}
    numkeys = len(reqkeys)
    error_group: str = "errors"
    direct_read: bool = True
    filename: str

    def __init__(self, filename: Union[str, os.PathLike], keep_open: bool = False):
        self.filename = filename
        if not os.path.isfile(self.filename):
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with h5py.File(self.filename, "w") as hf:
                pass
        self._hf = None
        if keep_open:
            self._hf = h5py.File(self.filename, "a")

    def _open(self, mode="r"):
        if self._hf is not None:
            return self._hf
        return h5py.File(self.filename, mode)

    def _close(self, hf):
        if hf is not self._hf:
            hf.close()

    def close(self):
        """Close the persistently held file handle, if any."""
        if self._hf is not None:
            self._hf.close()
            self._hf = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def key_from_hindex(hindex):
        """sdh"""
        pdb = hindex[0:4]
        return f"/_{pdb[1:3]}/_{pdb}/_{hindex}"

    # https://github.com/harvardnlp/botnet-detection/blob/master/graph_data_storage.md
    def write(self, emb, code, sdh=True):
        if not sdh:
            pdb, chain, hnum = code.split("_")
        hf = self._open(mode="a")
        try:
            hf.create_dataset(
                code,
                shape=emb.shape,
                data=emb.numpy(),
                dtype=np.float16,
                compression="gzip",
            )
        finally:
            self._close(hf)

    def read(self, code, sdh=True) -> torch.Tensor:
        if not sdh:
            pdb, chain, hnum = code.split("_")
        hf = self._open(mode="r")
        try:
            return torch.from_numpy(hf[code][:])
        finally:
            self._close(hf)

    @property
    def codes(self) -> list:
        hf = self._open(mode="r")
        try:
            return list(hf.keys())
        finally:
            self._close(hf)
