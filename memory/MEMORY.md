# Memory

- [Self-loops + virtual CB for glycine](2026-08-27-self-loops-and-virtual-cb.md) — Made self-loops explicit/idempotent at data creation in cc_calc.py & calc.py (reader must drop dgl.add_self_loop); added virtual_cb helper in numeric.py to fill glycine's NaN CB so distancemx is NaN-free.
