# Online RAS Bandit

`ras-thompson` learns a dataset-scale sampling policy from partial feedback.
Each observed sample contributes its current pooled masked-context latent. A
discounted Bayesian linear model predicts batch RAS from those contexts, and a
Thompson draw produces the next epoch's sampling probabilities.

The implementation does not rescore the full dataset with encoder forwards.
Contexts are collected from normal training forwards and retained in an EMA
cache. The reward is the aggregate batch RAS computed from the normal training
backward against a periodically refreshed richness gradient.

The sampler, cache, and reward normalizer are included in checkpoints.
