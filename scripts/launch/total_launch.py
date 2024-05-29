"""
That file starts full cycle of step-by-step execution of the project.
First, the necessary data for the polyhedra datasets is collected.
Then the models are trained and optimized using selection of hyperparameters
on all types of polyhedra descriptors and PointCloudDataset (just for PointNet).

The best configurations for each model with a specific type of polyhedra dataset are printed.
"""

from scripts.hyp_search.run_all_models import main as run_models
from data_massage.collector import main as run_collection

if __name__ == "__main__":
    run_collection()
    run_models()
