from .random_forests import random_forest_models
from .regressions import regressions_models
from .boost import boosting_models

all_models = {
  **regressions_models,
  **random_forest_models,
  **boosting_models

}