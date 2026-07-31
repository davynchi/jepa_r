# Official I-JEPA with residual-Q17

This directory contains the official `facebookresearch/ijepa` training code
with an optional centered residual-Q17 auxiliary objective.

The DataSphere experiment is fixed in
`configs/tiny_vits4_q17_matched_g5_ep300_b256_q64_datasphere.yaml`:

- Tiny ImageNet, all 100,000 training images;
- official ViT-S/4 and predictor depth 6, dimension 192;
- 300 epochs, main batch 256, BF16;
- uniform dataset sampling;
- matched residual-Q17 with gradient ratio 5%;
- Q17 sub-batch 64 and fixed flip, blur, and color transformations;
- checkpoints at epochs 50, 100, 150, 200, 250, and 300.

From the repository root in a DataSphere terminal or `%%bash` cell:

```bash
cd /home/jupyter/project/kostya/jepa_r
third_party/ijepa_q17/run_datasphere_q17.sh
```

DataSphere notebook shells do not support detached background processes, so
the launcher intentionally remains in the foreground. Outputs are written to
`outputs/official_ijepa_q17/cloud_tiny_vits4_q17_matched_g5_ep300_b256_q64/`.

To resume, set `meta.load_checkpoint: true` in the YAML. With
`meta.read_checkpoint: null`, training loads `jepa-latest.pth.tar` from the
same output directory and restores the model, both optimizers, scaler, and
Q17 EMA statistics.
