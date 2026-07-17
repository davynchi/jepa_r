# Vendored I-JEPA code

Selected files copied verbatim from https://github.com/facebookresearch/ijepa
(Meta Platforms, Inc.), the reference implementation of

> Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
> Predictive Architecture" (I-JEPA), CVPR 2023.

## License

The upstream code is licensed **CC BY-NC 4.0 (Attribution-NonCommercial)** —
see `LICENSE`. That is *not* an open-source license: it permits research and
other non-commercial use with attribution, and forbids commercial use. This
vendored copy is used here for academic research only.

## What is vendored, and why

| File | Purpose |
| --- | --- |
| `src/masks/multiblock.py` | `MaskCollator` — the multi-block masking strategy (one large context block, several smaller non-overlapping target blocks). This sampling logic is the part worth reusing exactly. |
| `src/masks/utils.py` | `apply_masks` — gathers the patch tokens named by a mask index tensor. |
| `in1k_vith14_ep300.yaml` | Upstream's reference hyperparameters, kept for reference only (ImageNet/ViT-H scale — not our regime). |

Their `src/models/vision_transformer.py` is deliberately **not** vendored: our
adaptation replaces the ViT encoder with a CNN (see below), so the ViT encoder
and its ViT predictor are not used.

## How our adaptation differs

Our code lives in `src/jepa/ijepa_spatial.py` and is a *reimplementation in
their spirit*, not a drop-in port:

- **CNN encoder instead of ViT.** A ViT ingests a variable-length set of
  visible patch tokens and mixes them with self-attention; a CNN cannot. Our
  encoder applies a small CNN to each patch independently and mean-pools the
  visible patches into one context summary. This keeps the anti-collapse
  mechanism (a predictor conditioned on *which* position it must predict) but
  loses the context's spatial structure — the predictor knows what to predict,
  not where the evidence came from.
- **MLP predictor with learned positional embeddings** instead of their
  transformer predictor with `mask_token` + fixed 2-D sin-cos embeddings.
- **Kept from upstream:** `F.layer_norm` over the feature dim of the target
  representation before the loss, and `smooth_l1_loss` — both taken directly
  from their `src/train.py`.
- **Scale:** 64x64 Shapes3D frames, patch size 8 (an 8x8 patch grid), versus
  their 224x224 / patch 14-16.
