# Supplementary material

The PDF is organized by the paper's claims rather than by raw output filename:

1. title, complete author list, artifact version, and release/DOI status;
2. all 40 attribution cells;
3. all 16 zero-gate/positive-Shapley cases with intervals and refit frequencies;
4. all 120 Rashomon coordinate intervals and the KKT-audit pointer;
5. nested system-disjoint results by corpus and budget;
6. all four null/control families and interaction-correlation uncertainty;
7. exact file mapping and interpretation limits.

Rebuild all public tables and the PDF with:

```bash
bash scripts/reproduce_analysis.sh
```

The PDF, compact tables, and `generated_full_tables.md` derive only from
committed CSVs. The CSVs remain canonical. Full re-extraction requires the
separately licensed datasets and model checkpoints.
