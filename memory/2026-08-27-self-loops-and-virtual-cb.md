# 2026-08-27 — Self-loops + virtual CB for glycine

## 1. Self-loops owned by data creation
**Problem:** DGL warned about self-loops. Investigation showed `read_struct_cc` (cc_calc.py) and `read_struct` (calc.py) already emit one self-loop per node *implicitly* — `ca_dist.fill_diagonal_(0)` makes the diagonal `< t`, so `th.where(ca_dist < t)` includes every `(i, i)`. The real bug: the downstream H5 reader (in a **separate training repo**, not graph_toolbox) also calls `dgl.add_self_loop(g, fill_data=0.01)`, and DGL 1.0.1's `add_self_loop` does **not** deduplicate → every node got **two** self-loops (one real-feature, one 0.01-fill). Confirmed empirically (edges 4→7, self-loops 3→6).

**Change (user chose "data creation owns them"):** in both `cc_calc.py` and `calc.py`, replaced `u, v = th.where(ca_dist < t)` with an explicit, idempotent guarantee:
```python
contact_mask = ca_dist < t
contact_mask.fill_diagonal_(True)
u, v = th.where(contact_mask)
```
Verified: exactly 1 self-loop/node, 0 missing, 0 duplicates, `is_self=1`, efeat dim 21.

**Action still required on the training-repo side:** remove `dgl.add_self_loop(g, fill_data=0.01)` from the reader. Existing H5 files already contain the loops, so no regeneration needed. (Self-loop edge features become the computed `is_self` row instead of the 0.01 constant.)

## 2. Virtual CB atoms for glycine (cc_calc.py)
**Problem:** Glycine has no CB, so cc_calc appended a NaN placeholder → NaN propagated into `cb_dist` → `distancemx` channel 1 (corrupting the CB-distance target for every glycine).

**Change:**
- Added `virtual_cb(n, ca, c)` `@th.jit.script` helper in `numeric.py` — idealized CB from backbone via the standard trRosetta/RGN tetrahedral constants (`-0.58273431*a + 0.56802827*b - 0.54067466*c + ca`, `a = cross(ca-n, c-ca)`), a single vectorized cross-product.
- In `read_struct_cc`, after coordinate tensors are built and before `cb_dist`, replaced only glycine (NaN) CB rows: `res_cb = th.where(gly_mask, virtual_cb(...), res_cb)` where `gly_mask = th.isnan(res_cb).any(dim=-1, keepdim=True)`.

**Verified:** `distancemx` CB-channel NaN count 40+/31 rows → **0** (1xyz/6iii). Formula reproduces real CB within **0.10 Å mean / 0.35 Å max** over 600 residues.

**Note:** glycine CG/CD remain NaN (no γ carbon), so its side-chain dihedrals/rotation frame stay NaN — expected. Scope was cc_calc.py only; calc.py has the identical glycine-NaN pattern if mirroring is wanted later.

## Test status
`pytest tests/test_feature.py` → 12 failures, all **pre-existing** (unimplemented `with_interactions=True` guard at calc.py:94), identical on committed HEAD. No new regressions from either change.
