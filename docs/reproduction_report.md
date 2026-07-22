# Faithful Reproduction of Gohr's Speck32/64 Neural Distinguisher

## Objective

This repository contains a faithful reproduction of the five-round Speck32/64 neural distinguisher introduced by Gohr (CRYPTO 2019).

The objective of this work is **not** to improve the original model, but to establish a verified baseline that can be used for subsequent auditing, benchmarking, and scientific analysis of AI-based cryptanalysis.

---

## Methodology

The original implementation was reproduced using the published DeepSpeck codebase.

The training configuration was preserved:

- Speck32/64
- 5-round distinguisher
- 10 residual blocks
- Adam optimizer
- Cyclic learning-rate schedule
- Mean Squared Error loss
- 200 epochs
- Online synthetic training data generation

Only compatibility updates required for TensorFlow 2.20 and Keras 3 were applied.

---

## Results

Training completed successfully.

Final validation accuracy:

0.9291

This result is consistent with the expected performance reported for the original implementation.

---

## Repository Contents
freshly_trained_nets/

├── best5depth10.h5
├── h5r_depth10.npy
└── hist5r_depth10.p

notebooks/

└── gohr_reproduction_tf2.ipynb

docs/

├── training_configuration.md
├── reproduction_environment.md
└── reproduction_report.md

---

## Reproducibility

Git branch:

reproduction-tf2

Git tag:

gohr-faithful-reproduction

Commit:

2b54425

---

## Future Work

This repository serves as the frozen baseline for future work involving:

- auditing dataset leakage
- representation analysis
- robustness evaluation
- benchmarking AI-assisted cryptanalysis
- reproducibility studies

No experimental modifications should be made directly on this branch.