# Generated public tables

## Nested system-disjoint selection

| Rule | Coalition scores | Mean MSE/full | Mean rank | Best/tied |
|---|---:|---:|---:|---:|
| Gate weight | 1 | 1.0834 | 2.375 | 4/16 |
| Singleton MSE | 10 | 1.0786 | 2.250 | 7/16 |
| Exact Shapley | 1,023 | 1.0792 | 2.062 | 7/16 |
| Inner search | 1,023 | 1.0771 | 2.000 | 9/16 |

## Macro bootstrap contrasts

| method    | reference    |   mean_normalized_mse_delta |   ci95_low |   ci95_high |   probability_delta_lt_0 |   n_corpora |   n_budgets |   n_bootstrap |   p_two_sided_bootstrap |   p_holm_all_five |
|:----------|:-------------|----------------------------:|-----------:|------------:|-------------------------:|------------:|------------:|--------------:|------------------------:|------------------:|
| gate      | inner_search |                    0.006232 |   0.000279 |    0.011473 |                 0.020800 |           4 |           4 |         10000 |                0.041600 |          0.208000 |
| shapley   | inner_search |                    0.002016 |  -0.009829 |    0.017517 |                 0.387200 |           4 |           4 |         10000 |                0.774400 |          1.000000 |
| singleton | inner_search |                    0.001424 |  -0.008847 |    0.010178 |                 0.366800 |           4 |           4 |         10000 |                0.733600 |          1.000000 |
| gate      | singleton    |                    0.004808 |  -0.006231 |    0.016102 |                 0.213600 |           4 |           4 |         10000 |                0.427200 |          1.000000 |
| shapley   | singleton    |                    0.000593 |  -0.008386 |    0.013577 |                 0.472100 |           4 |           4 |         10000 |                0.944200 |          1.000000 |

## Zero-gate / positive-Shapley cases

| dataset   | feature      |   gate_weight |   leave_one_out_mse_utility |   exact_shapley_mse_utility | gate_zero_lt_1e_6   | gate_zero_positive_shapley   | loo_signed_le_1e_6_positive_shapley   | loo_absolute_le_1e_6_positive_shapley   |
|:----------|:-------------|--------------:|----------------------------:|----------------------------:|:--------------------|:-----------------------------|:--------------------------------------|:----------------------------------------|
| brspeech  | beats        |      0.000000 |                   -0.000000 |                    0.048849 | True                | True                         | True                                  | True                                    |
| brspeech  | auditory_erb |      0.000000 |                   -0.000000 |                    0.051379 | True                | True                         | True                                  | True                                    |
| brspeech  | speaker      |      0.000000 |                   -0.000000 |                    0.120166 | True                | True                         | True                                  | True                                    |
| brspeech  | rmvpe_cont   |      0.000000 |                    0.000000 |                    0.004173 | True                | True                         | True                                  | True                                    |
| brspeech  | ced          |      0.000000 |                    0.000000 |                    0.124296 | True                | True                         | True                                  | True                                    |
| bvcc      | auditory_erb |      0.000000 |                   -0.000000 |                    0.023156 | True                | True                         | True                                  | True                                    |
| bvcc      | rmvpe_cont   |      0.000000 |                   -0.000000 |                    0.006789 | True                | True                         | True                                  | True                                    |
| bvcc      | rmvpe_quant  |      0.000000 |                   -0.000000 |                    0.003386 | True                | True                         | True                                  | True                                    |
| bvcc      | egemaps      |      0.000000 |                   -0.000000 |                    0.049994 | True                | True                         | True                                  | True                                    |
| singmos   | auditory_erb |      0.000000 |                   -0.000000 |                    0.018057 | True                | True                         | True                                  | True                                    |
| singmos   | rmvpe_cont   |      0.000000 |                    0.000000 |                    0.011110 | True                | True                         | True                                  | True                                    |
| singmos   | rmvpe_quant  |      0.000000 |                    0.000000 |                    0.002701 | True                | True                         | True                                  | True                                    |
| tmhintqi  | auditory_erb |      0.000000 |                    0.000000 |                    0.022157 | True                | True                         | True                                  | True                                    |
| tmhintqi  | rmvpe_cont   |      0.000000 |                    0.000000 |                    0.002343 | True                | True                         | True                                  | True                                    |
| tmhintqi  | rmvpe_quant  |      0.000000 |                    0.000000 |                    0.003211 | True                | True                         | True                                  | True                                    |
| tmhintqi  | egemaps      |      0.000000 |                    0.000000 |                    0.069490 | True                | True                         | True                                  | True                                    |

## Player-partition sensitivity

| dataset   | scenario                   |   n_players |   zero_gate_positive_shapley_count |   group_rank_rho_vs_original_aggregated |   owen_rank_rho_vs_original |   owen_sign_agreement |   efficiency_error |   game_total_mse_utility |
|:----------|:---------------------------|------------:|-----------------------------------:|----------------------------------------:|----------------------------:|----------------------:|-------------------:|-------------------------:|
| brspeech  | original_10                |          10 |                                  5 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.684177 |
| brspeech  | rmvpe_grouped_9            |           9 |                                  5 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.684177 |
| brspeech  | prosody_grouped_8          |           8 |                                  4 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.684177 |
| brspeech  | remove_rmvpe_quant_9       |           9 |                                  5 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.684177 |
| brspeech  | remove_egemaps_9           |           9 |                                  2 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.668976 |
| brspeech  | remove_quant_and_egemaps_8 |           8 |                                  2 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.668976 |
| bvcc      | original_10                |          10 |                                  4 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.637535 |
| bvcc      | rmvpe_grouped_9            |           9 |                                  3 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.637535 |
| bvcc      | prosody_grouped_8          |           8 |                                  2 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.637535 |
| bvcc      | remove_rmvpe_quant_9       |           9 |                                  3 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.637535 |
| bvcc      | remove_egemaps_9           |           9 |                                  3 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.637535 |
| bvcc      | remove_quant_and_egemaps_8 |           8 |                                  2 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.637535 |
| singmos   | original_10                |          10 |                                  3 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.240374 |
| singmos   | rmvpe_grouped_9            |           9 |                                  2 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.240374 |
| singmos   | prosody_grouped_8          |           8 |                                  1 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.240374 |
| singmos   | remove_rmvpe_quant_9       |           9 |                                  2 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.240374 |
| singmos   | remove_egemaps_9           |           9 |                                  2 |                                1.000000 |                    1.000000 |              0.888889 |           0.000000 |                 0.228660 |
| singmos   | remove_quant_and_egemaps_8 |           8 |                                  1 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.228660 |
| tmhintqi  | original_10                |          10 |                                  4 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.660279 |
| tmhintqi  | rmvpe_grouped_9            |           9 |                                  3 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.660279 |
| tmhintqi  | prosody_grouped_8          |           8 |                                  2 |                                0.976190 |                    1.000000 |              1.000000 |           0.000000 |                 0.660279 |
| tmhintqi  | remove_rmvpe_quant_9       |           9 |                                  3 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.660279 |
| tmhintqi  | remove_egemaps_9           |           9 |                                  3 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.660279 |
| tmhintqi  | remove_quant_and_egemaps_8 |           8 |                                  2 |                                1.000000 |                    1.000000 |              1.000000 |           0.000000 |                 0.660279 |
