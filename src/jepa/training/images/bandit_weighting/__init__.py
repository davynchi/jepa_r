"""Online bandit samplers for spatial I-JEPA data weighting."""

from jepa.training.images.bandit_weighting.context_cache import LatentContextCache
from jepa.training.images.bandit_weighting.linear_thompson import (
    DiscountedLinearThompsonSampler,
    LinearThompsonConfig,
)
from jepa.training.images.bandit_weighting.online_ras import (
    RichnessGradientSnapshot,
    batch_ras_from_parameter_gradients,
    capture_richness_gradient,
)
from jepa.training.images.bandit_weighting.reward_normalizer import (
    DiscountedRewardNormalizer,
)

__all__ = [
    "DiscountedLinearThompsonSampler",
    "DiscountedRewardNormalizer",
    "LatentContextCache",
    "LinearThompsonConfig",
    "RichnessGradientSnapshot",
    "batch_ras_from_parameter_gradients",
    "capture_richness_gradient",
]
