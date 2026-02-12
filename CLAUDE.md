# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**graph_toolbox** is a Python library for converting PDB (Protein Data Bank) structures into graph representations for graph neural networks. It computes geometric and chemical features (distances, dihedral angles, backbone rotations, residue-residue interactions) and outputs DGL graphs.

## Build & Test Commands

```bash
# Install package
pip install -e .

# Run all tests
make run-tests
# or equivalently:
python -m pytest tests/test_feature.py

# Run a single test
python -m pytest tests/test_feature.py::test_read_struct -v
python -m pytest tests/test_feature.py -k "test_graph_data" -v
```

## Architecture

### Data Pipeline

```
PDB File → biopandas DataFrame → read_struct() → StructFeats → GraphData → DGL Graph
                                                                    ↕
                                                              HDF5 / .pth file
```

### Key Modules

- **`feature/base.py`** — `GraphData` class: main data structure holding node/edge features, sequences, distance matrices, and rotation matrices. Entry points: `GraphData.from_pdb()`, `GraphData.from_h5()`, `.to_dgl()`, `.to_h5()`.
- **`feature/calc.py`** — `read_struct()`: parses PDB DataFrames, extracts backbone/sidechain atoms per residue, computes CA-CA distances, dihedral angles, and interaction edges.
- **`feature/numeric.py`** — JIT-compiled (`@th.jit.script`) geometry functions: pairwise distances, backbone dihedrals (phi/psi/omega), sidechain dihedrals (chi1/chi2).
- **`feature/rotary_matrix.py`** — Backbone rotation matrix calculations: local coordinate frames from N/CA/C atoms, pairwise relative rotations.
- **`feature/models.py`** — `StructFeats` pydantic dataclass: intermediate container between `read_struct()` and `GraphData`.
- **`feature/params.py`** — Constants: amino acid mappings, atom selection tables per residue type, interaction distance thresholds.
- **`feature/dataset.py`** — `H5Handle` / `EmbH5Handle`: HDF5 I/O for batch storage and retrieval of graph data.

### Feature Dimensions

**Edge features (11):** disulfide, hydrophobic, cation_pi, arg_arg, salt_bridge, hbond, vdw, self, is_seq, is_seq_not, is_struct

**Node features (5):** phi, psi, omega (backbone dihedrals), chi1, chi2 (sidechain dihedrals)

### Supporting Modules

- **`parse/`** — PDB parsing utilities (atomium integration, coordinate extraction)
- **`inout/`** — File I/O helpers (JSON, pickle, gzip)
- **`ops/blockdiag.py`** — Sparse block diagonal matrix operations for batched graphs
- **`plot/`** and **`pymol/`** — Visualization utilities

## Key Dependencies

- `torch` — tensors, JIT compilation
- `dgl==1.1.3` — graph neural network framework
- `biopandas` — PDB file parsing
- `pydantic` — data validation for StructFeats

## Test Data

Test PDB files are in `tests/data/` (1xyz.pdb, 6iii.pdb, pdb4err.ent). Tests parametrize over multiple CA-CA distance thresholds (5, 7, 9 Å) and validate feature shapes and interaction flags.
