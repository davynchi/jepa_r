import numpy as np

from scripts.analysis.evaluate_factorization_proxy_transfer import (
    fit_affine,
    predict_affine,
    prediction_metrics,
)


def test_affine_transfer_recovers_linear_relation() -> None:
    feature = np.asarray([1.0, 2.0, 3.0, 4.0])
    target = 0.2 + 0.5 * feature
    coefficients = fit_affine(feature, target)
    predicted = predict_affine(coefficients, feature)
    metrics = prediction_metrics(target, predicted)

    assert np.allclose(coefficients, [0.2, 0.5])
    assert metrics["mae"] < 1e-12
    assert metrics["r2"] > 0.999
