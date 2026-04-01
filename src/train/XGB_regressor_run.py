rom spatialmt.config.paths import Dirs

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from tqdm import tqdm

# callback to track progress
class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, n_estimators):
        self.pbar = tqdm(total=n_estimators, desc="Training", unit="tree")

    def after_iteration(self, model, epoch, evals_log):
        train_loss = list(evals_log.get("train", {}).values())
        val_loss   = list(evals_log.get("eval",  {}).values())
        postfix = {}
        if train_loss:
            postfix["train_loss"] = f"{train_loss[0][-1]:.4f}"
        if val_loss:
            postfix["val_loss"] = f"{val_loss[0][-1]:.4f}"
        self.pbar.set_postfix(postfix)
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model
    
if __name__ == "__main__":
    data_dir = Dirs.model_data_ml
    
    cell_labels = pd.read_csv(data_dir / "cell_labels.csv").squeeze()
    gene_labels = pd.read_csv(data_dir / "gene_labels.csv").squeeze()
    y_raw = pd.read_csv(data_dir / "pseudotime_labels.csv").squeeze()
    
    X = pd.DataFrame(
        np.load(data_dir / "expression_matrix.npz")["X"],
        index=cell_labels,
        columns=gene_labels,
    )