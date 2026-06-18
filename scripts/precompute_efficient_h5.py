"""
One-time preprocessing script that converts graphs_data.h5 into graphs_data.efficient.h5.

The efficient file stores:
  - efeats_flat  [E, D]      — edge features indexed by (u, v), replaces the NxN efeats matrix
  - target       [N, N, 19]  — distancemx + backbone_relrot (flat) + sidechain_relrot (flat) pre-concatenated
  - u, v, sequence, nfeats, segment_id — copied unchanged

Run once per dataset:
    python scripts/precompute_efficient_h5.py -dataset /path/to/dataset
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def iter_leaf_groups(h5file):
    """Yield (key, group) for every leaf group (helix-level) in the file."""

    def _visit(name, obj):
        if isinstance(obj, h5py.Group) and len(list(obj.keys())) > 0:
            # leaf group: has datasets, not only sub-groups
            children = list(obj.values())
            if any(isinstance(c, h5py.Dataset) for c in children):
                yield name, obj

    results = []
    h5file.visititems(
        lambda name, obj: (
            results.append((name, obj))
            if (isinstance(obj, h5py.Group) and any(isinstance(c, h5py.Dataset) for c in obj.values()))
            else None
        )
    )
    return results


def precompute(dataset_path: Path):
    graphs_dir = dataset_path / "graphs"
    src_path = graphs_dir / "graphs_data.h5"
    dst_path = graphs_dir / "graphs_data.efficient.h5"

    if not src_path.is_file():
        raise FileNotFoundError(f"Source not found: {src_path}")

    print(f"Source : {src_path}")
    print(f"Output : {dst_path}")

    with h5py.File(src_path, "r", swmr=True) as src, h5py.File(dst_path, "a") as dst:
        leaf_groups = iter_leaf_groups(src)
        print(f"Groups to process: {len(leaf_groups)}")

        REQUIRED_KEYS = ("u", "v", "sequence", "nfeats", "segment_id", "efeats_flat", "target")
        skipped = 0
        for key, in_grp in tqdm(leaf_groups, unit="helix"):
            if key in dst:
                missing = [k for k in REQUIRED_KEYS if k not in dst[key]]
                if missing:
                    raise KeyError(f"Incomplete group '{key}' in dst — missing datasets: {missing}. Delete the efficient H5 and rerun.")
                skipped += 1
                continue

            # ensure parent groups exist
            dst.require_group(str(Path(key).parent))
            out_grp = dst.require_group(key)

            u = in_grp["u"][:]
            v = in_grp["v"][:]

            # pre-indexed edge features: [E, D]
            efeats_flat = in_grp["efeats"][:][u, v]
            out_grp.create_dataset("efeats_flat", data=efeats_flat.astype(np.float32), compression="lzf")

            # pre-concatenated target: [N, N, 19]
            d = in_grp["distancemx"][:]  # [N, N, 1] or [N, N, 2]
            br = in_grp["backbone_relrot"][:].reshape(d.shape[0], d.shape[1], -1)  # [N, N, 9]
            sr = in_grp["sidechain_relrot"][:].reshape(d.shape[0], d.shape[1], -1)  # [N, N, 9]
            target = np.concatenate([d, br, sr], axis=-1).astype(np.float32)
            out_grp.create_dataset("target", data=target, compression="lzf")

            for k in ("u", "v", "sequence", "nfeats", "segment_id"):
                out_grp.create_dataset(k, data=in_grp[k][:], compression="lzf")

        if skipped:
            print(f"Skipped {skipped} already-processed groups (idempotent run).")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, required=True, help="Dataset root directory")
    args = parser.parse_args()
    precompute(Path(args.dataset))
