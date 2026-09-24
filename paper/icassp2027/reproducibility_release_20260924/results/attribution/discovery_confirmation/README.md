# Validation-discovery / test-confirmation attribution audit

The frozen coalition gates were fitted only on system-grouped OOF predictions. Cases were discovered using validation Shapley utility and full-gate weight, then the list was frozen before computing any test confirmation value. The test table therefore contains no post-hoc search for new positive cases.

Discovery rule: `w < 1e-06` and `phi_val > 0`. The run found 16 validation cases; 15 retained positive point utility on test and 11 had a positive 95% system-bootstrap lower bound.

Files: `validation_discovered_cases.csv` (frozen list), `test_confirmation_of_validation_cases.csv` (only frozen cases with intervals), `discovery_confirmation_principal.csv` (joined paper table), and `test_all_cases_confirmation_audit.csv` (audit-only complete test values; do not use it to redefine the list).
