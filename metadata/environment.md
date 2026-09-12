# Reference environment and compute record

- Python 3.10.12
- NumPy 2.2.6
- pandas 2.3.3
- SciPy 1.15.3
- scikit-learn 1.7.2
- PyTorch 2.5.1+cu118
- Matplotlib 3.10.0 for public tables/supplement generation

All paper-level coalition, bootstrap, and nested-selection analyses used CPU.
Representation extraction used a 16 GB NVIDIA GPU sequentially to avoid OOM.
The archived nested system-disjoint run started at 00:17:22 and completed at
00:20:59 UTC on 2026-09-12 (about 3.6 minutes, including solver parity and
10,000 bootstrap replicates). This timing is descriptive, not a portable
benchmark.

