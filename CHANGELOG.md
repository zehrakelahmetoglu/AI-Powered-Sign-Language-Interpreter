# Changelog

## 2026-04-20 - Dataset Health Snapshot

### Current Scores

- Total classes: 255
- Total samples: 291
- Samples per class (min/median/mean/max): 1 / 1.0 / 1.14 / 3
- Quantity score: 1.14 / 100
- Coverage >=20 score: 0.00 / 100
- Coverage >=50 score: 0.00 / 100
- Balance score: 89.11 / 100
- Min support score: 10.00 / 100
- Dataset Health Score: 2.07 / 100 (Grade: F)

### Gap To Target

- Needed to reach >=20 samples per class: 4809
- Needed to reach >=50 samples per class: 12459
- Needed to reach >=100 samples per class: 25209

### Why The Score Is Low

- Quantity is very low: average sample count per class is only 1.14.
- Coverage thresholds are not met: no class reaches 20 or 50 samples, so both coverage dimensions are 0.
- Minimum support is weak: the weakest classes have only 1 sample, which is far below the bootstrap floor.
- Distribution is relatively balanced, but balanced scarcity is still scarcity; classes being equally low does not provide enough training signal.

### Practical Interpretation

- The dataset is not yet ready for robust multi-class training across 255 classes.
- Most likely outcome at this stage is memorization and poor generalization.
- Recommended immediate strategy is phased scope reduction (core classes first) and aggressive data growth to at least 20 per active class.
