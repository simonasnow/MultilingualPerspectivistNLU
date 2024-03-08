from config import config
from perspectivist_experiments import datasets, training_models, performance

# datasets.create_dataset(config)
training_models.get_predictions(config)
# performance.get_results(config)