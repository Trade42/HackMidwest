"""Training-only preprocessing for bounded circuit and controlled features."""

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Analytic expectations can contain roundoff around zero. StandardScaler's
# relative constant detection can scale that ~1e-17 noise to unit variance.
# This absolute floor removes std <= 1e-12 before scaling. It is a numerical
# tolerance for our bounded features, not a validation-tuned hyperparameter.
FEATURE_VARIANCE_FLOOR = 1e-24


def make_feature_classifier(estimator):
    """Fit both feature selection and scaling on classifier training rows only."""
    return make_pipeline(
        VarianceThreshold(threshold=FEATURE_VARIANCE_FLOOR),
        StandardScaler(),
        estimator,
    )
