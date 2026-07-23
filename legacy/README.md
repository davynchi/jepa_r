# Legacy spatial I-JEPA

This directory contains the pre-upstream patch-CNN implementation and its
corrupted-Shapes3D weighting benchmark.

The implementation is retained only for controlled comparisons. It encodes
patches independently, uses batch-shared masks, and predicts targets from a
mean-pooled context summary. It must not be confused with the upstream-style
I-JEPA implementation under `src/jepa` and `scripts/`.

Entry points:

- `train_shapes3d_corrupted.py`: one training run.
- `track_shapes3d_corrupted.py`: clean held-out diagnostics.
- `train_shapes3d_corrupted_with_diagnostics.sh`: training plus watcher.
- `run_shapes3d_corrupted_suite.sh`: two-GPU comparison suite.
