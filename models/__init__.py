from .random_forests import random_forest_models
from .regressions import regressions_models
from .boost import boosting_models
from .primary import primary_models
from .regressors import regressor_comparisons
from .pca import pca_models
from .knn import knn

all_models = {
  **regressions_models,
  **random_forest_models,
  **boosting_models,
  **knn,
  **primary_models,
  **regressor_comparisons,
  **pca_models
}