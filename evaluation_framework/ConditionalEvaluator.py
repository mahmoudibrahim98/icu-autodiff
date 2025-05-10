'''
Code for GRU classifier to predict slice label from raw features
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n  = self.rnn(x)             
        last    = h_n[-1]                 
        logits  = self.fc(last)           
        return logits                     

class GRUClassifier:
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 lr=1e-3):
        self.model     = GRUModel(input_size, hidden_size, num_layers, num_classes)
        self.model     = nn.DataParallel(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self,
                    train_x, train_y,
                    val_x=None, val_y=None,
                    epochs=50,
                    batch_size=128,
                    patience=5,
                    min_delta=1e-4,
                    verbose=True):
        """
        - train_x: (N, seq_len, feat)
        - train_y: (N,) integer labels
        - val_x, val_y: optional validation split
        - patience: #epochs with no val_loss improvement before stop
        - min_delta: required decrease in val_loss to reset patience
        """
        # convert to tensors & loaders
        def make_loader(X, Y):
            if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
            if isinstance(Y, np.ndarray): Y = torch.from_numpy(Y).long()
            ds = TensorDataset(X, Y)
            return DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        train_loader = make_loader(train_x, train_y)
        if val_x is not None and val_y is not None:
            val_loader = make_loader(val_x, val_y)
        else:
            val_loader = None

        best_val_loss = float('inf')
        best_epoch    = 0

        for epoch in range(1, epochs+1):
            # --- Training pass ---
            self.model.train()
            train_loss = 0.
            for X_mb, y_mb in train_loader:
                X_mb, y_mb = X_mb.to(device), y_mb.to(device)
                self.optimizer.zero_grad()
                logits = self.model(X_mb)
                loss   = self.criterion(logits, y_mb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X_mb.size(0)
            train_loss /= len(train_loader.dataset)

            # --- Validation pass ---
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.
                with torch.no_grad():
                    for X_mb, y_mb in val_loader:
                        X_mb, y_mb = X_mb.to(device), y_mb.to(device)
                        logits     = self.model(X_mb)
                        loss       = self.criterion(logits, y_mb)
                        val_loss  += loss.item() * X_mb.size(0)
                val_loss /= len(val_loader.dataset)

                # Check improvement
                improved = (best_val_loss - val_loss) > min_delta
                if improved:
                    best_val_loss = val_loss
                    best_epoch    = epoch
                elif (epoch - best_epoch) >= patience:
                    if verbose:
                        print(f"⏱ Early stopping at epoch {epoch} "
                              f"(no val_loss improvement in {patience} epochs)")
                    break

                if verbose:
                    print(f"Epoch {epoch:02d} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Best: {best_val_loss:.4f} (ep {best_epoch})")
            else:
                if verbose:
                    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}")

    def evaluate(self, test_x, test_y):
        """Returns overall accuracy on test set."""
        if isinstance(test_x, np.ndarray):
            test_x = torch.from_numpy(test_x).float()
        if isinstance(test_y, np.ndarray):
            test_y = torch.from_numpy(test_y).long()

        ds = TensorDataset(test_x, test_y)
        loader = DataLoader(ds, batch_size=256)

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_mb, y_mb in loader:
                X_mb = X_mb.to(device)
                logits = self.model(X_mb)
                preds  = logits.argmax(dim=1).cpu()
                all_preds .append(preds)
                all_labels.append(y_mb)

        y_pred  = torch.cat(all_preds).numpy()
        y_true  = torch.cat(all_labels).numpy()
        return accuracy_score(y_true, y_pred)
    def evaluate_detailed(self, test_x, test_y, class_names=None):
        """
        Runs the model on test_x/test_y and prints:
          - overall accuracy
          - confusion matrix
          - per-class precision, recall, f1
        Returns (accuracy, cm, report_dict).
        """
        # 1) prepare tensors & loader
        if isinstance(test_x, np.ndarray):
            test_x = torch.from_numpy(test_x).float()
        if isinstance(test_y, np.ndarray):
            test_y = torch.from_numpy(test_y).long()
        ds = TensorDataset(test_x, test_y)
        loader = DataLoader(ds, batch_size=256)

        # 2) collect all preds & labels
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_mb, y_mb in loader:
                X_mb = X_mb.to(device)
                logits = self.model(X_mb)               # [B, C]
                preds  = logits.argmax(dim=1).cpu()     # [B]
                all_preds.append(preds)
                all_labels.append(y_mb)

        y_pred  = torch.cat(all_preds).numpy()
        y_true  = torch.cat(all_labels).numpy()

        # 3) compute metrics
        acc = accuracy_score(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred)
        # as text report (string) or dict; here we get a dict
        report_dict = classification_report(
            y_true, 
            y_pred, 
            target_names=class_names, 
            output_dict=True
        )

        # 4) print human-readable
        print(f"Overall accuracy: {acc:.4f}\n")
        print("Confusion matrix:")
        print(cm, "\n")
        print("Classification report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

        return acc, cm, report_dict
    
    
    

'''
Code for Multi-task LSTM classifier to predict multiple categorical outputs from time series input data.
Each task has its own classification head, and the model is trained to minimize the
combined cross-entropy loss across all tasks.

tasks_info : dict
    Dictionary mapping task names to number of classes for each task.
    Example: {'diagnosis': 10, 'mortality': 2}
        

'''
    
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_auc_score, 
    log_loss, 
    confusion_matrix
)
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiTaskLSTM(nn.Module):
    """
    LSTM backbone + multiple classification heads
    (one for each task).
    """
    def __init__(self, input_size, hidden_size, num_layers, tasks_info):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Create one linear "head" for each task.
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, n_classes)
            for task, n_classes in tasks_info.items()
        })

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return {task: head(last_hidden) for task, head in self.heads.items()}


class MultiTaskDataset(Dataset):
    """
    Dataset wrapper for multi-task time-series data.

    Each sample returns the input sequence and a dict of labels per task.
    """
    def __init__(self, x_data, y_dict):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_dict = {
            task: torch.tensor(labels, dtype=torch.long)
            for task, labels in y_dict.items()
        }

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_i = self.x_data[idx]
        y_i = {task: labels[idx] for task, labels in self.y_dict.items()}
        return x_i, y_i


class MultiOutputTrainer:
    """
    Trainer class for multi-task LSTM models with optional early stopping.
    """
    def __init__(self, input_size, hidden_size, num_layers, tasks_info, lr=1e-3):
        self.model = MultiTaskLSTM(input_size, hidden_size, num_layers, tasks_info).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.tasks_info = tasks_info

    def train_model(
        self,
        train_x, train_y_dict,
        val_x=None, val_y_dict=None,
        epochs=10, batch_size=128,
        patience=3, min_delta=1e-4,
        verbose=True
    ):
        """
        Train with optional early stopping on validation loss.
        Args:
            patience (int): epochs to wait for improvement
            min_delta (float): minimum loss decrease to count
        """
        # Data loaders
        train_ds = MultiTaskDataset(train_x, train_y_dict)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = None
        if val_x is not None and val_y_dict is not None:
            val_ds = MultiTaskDataset(val_x, val_y_dict)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, epochs+1):
            # Training
            self.model.train()
            train_loss = 0.0
            for x_mb, y_mb in train_loader:
                x_mb = x_mb.to(device)
                logits = self.model(x_mb)
                loss = sum(
                    self.criterion(logits[task], y_mb[task].to(device))
                    for task in self.tasks_info
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * x_mb.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x_mb, y_mb in val_loader:
                        x_mb = x_mb.to(device)
                        logits = self.model(x_mb)
                        loss = sum(
                            self.criterion(logits[task], y_mb[task].to(device))
                            for task in self.tasks_info
                        )
                        val_loss += loss.item() * x_mb.size(0)
                val_loss /= len(val_loader.dataset)

                improved = (best_val_loss - val_loss) > min_delta
                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch
                elif (epoch - best_epoch) >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no improvement in {patience} epochs)")
                    break

                if verbose:
                    print(f"Epoch {epoch}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f} "
                          f"(Best: {best_val_loss:.4f} at ep {best_epoch})")
            else:
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")

    def evaluate(self, test_x, test_y_dict, batch_size=128):
        """
        Evaluate task-wise classification accuracy on test data.

        Args:
            test_x (np.ndarray): test inputs [N, seq_len, input_size].
            test_y_dict (dict[str,np.ndarray]): test labels for each task.
            batch_size (int): samples per batch.

        Returns:
            dict[str,float]: accuracy for each task.
        """
        test_ds = MultiTaskDataset(test_x, test_y_dict)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        self.model.eval()
        correct = {task: 0 for task in self.tasks_info}
        total = 0
        with torch.no_grad():
            for x_mb, y_mb in test_loader:
                x_mb = x_mb.to(device)
                logits = self.model(x_mb)
                total += x_mb.size(0)
                for task in self.tasks_info:
                    preds = logits[task].argmax(dim=1).cpu()
                    correct[task] += (preds == y_mb[task]).sum().item()
        for task, count in correct.items():
            print(f"{task} Accuracy: {count/total:.4f}")
        return {task: count/total for task, count in correct.items()}

    def evaluate_all_metrics(self, test_x, test_y_dict, batch_size=128):
        """
        Computes for each task:
          - accuracy
          - per-class precision, recall, F1
          - log-loss
          - ROC AUC (if binary)
          - confusion matrix
        Returns a dict of dicts, metrics[task_name][metric_name] = value.
        """
        # Prepare data loader
        if isinstance(test_x, np.ndarray):
            test_x = torch.from_numpy(test_x).float()
        # assume test_y_dict values are numpy arrays
        # we’ll collect true labels and raw logits
        all_logits = {t: [] for t in self.tasks_info}
        all_true   = {t: [] for t in self.tasks_info}

        ds = TensorDataset(
            test_x, 
            *[torch.from_numpy(test_y_dict[t]).long() for t in self.tasks_info]
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x_mb = batch[0].to(device)
                logits = self.model(x_mb)  # dict of [B, C_t]
                # store per-task
                for i, task in enumerate(self.tasks_info):
                    y_true = batch[i+1].numpy()
                    all_true[task].append(y_true)
                    all_logits[task].append(logits[task].cpu())

        metrics = {}
        for task, n_classes in self.tasks_info.items():
            # concatenate
            y_true    = np.concatenate(all_true[task])
            logits_t  = torch.cat(all_logits[task], dim=0)
            probs_t   = F.softmax(logits_t, dim=1).numpy()
            y_pred    = np.argmax(probs_t, axis=1)

            m = {}
            # 1) accuracy
            m['accuracy'] = accuracy_score(y_true, y_pred)

            # 2) classification report
            report = classification_report(
                y_true, y_pred, output_dict=True
            )
            m['precision_macro'] = report['macro avg']['precision']
            m['recall_macro']    = report['macro avg']['recall']
            m['f1_macro']        = report['macro avg']['f1-score']
            m['precision_weighted'] = report['weighted avg']['precision']
            m['recall_weighted']    = report['weighted avg']['recall']
            m['f1_weighted']        = report['weighted avg']['f1-score']

            # 3) log-loss
            m['log_loss'] = log_loss(y_true, probs_t)

            # 4) ROC AUC if binary
            if n_classes == 2:
                # take probability of class 1
                m['roc_auc'] = roc_auc_score(y_true, probs_t[:,1])

            # 5) confusion matrix
            m['confusion_matrix'] = confusion_matrix(y_true, y_pred)

            metrics[task] = m

        return metrics
    
    
import os


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split



def cross_validate_metrics(
    X, y_dict,
    input_size, hidden_size, num_layers, tasks_info,
    folds=5, random_state=42,
    lr=1e-3, epochs=10, batch_size=128,
    patience=3, min_delta=1e-4, verbose=False
):
    """
    Perform K-fold cross-validation, training with early stopping,
    and compute metrics with 95% CI.

    Args:
        X (np.ndarray): input data [N, seq_len, input_size]
        y_dict (dict[str, np.ndarray]): labels per task, each of shape [N]
        input_size, hidden_size, num_layers, tasks_info: model specs
        folds (int): number of CV splits
        random_state (int): seed for reproducibility
        lr, epochs, batch_size, patience, min_delta, verbose: training params

    Returns:
        df_folds (DataFrame): metrics for each fold (index=fold)
        summary (DataFrame): mean & 95% CI per metric across folds
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_metrics_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        # Split data
        X_full_train, X_test = X[train_idx], X[test_idx]
        y_full_train = {t: arr[train_idx] for t, arr in y_dict.items()}
        y_test = {t: arr[test_idx] for t, arr in y_dict.items()}

        # Further split training into train/val for early stopping
        train_index, val_index = train_test_split(
            np.arange(len(train_idx)),
            test_size=0.2,
            random_state=random_state
        )
        X_train = X_full_train[train_index]
        X_val   = X_full_train[val_index]
        y_train = {t: arr[train_index] for t, arr in y_full_train.items()}
        y_val   = {t: arr[val_index] for t, arr in y_full_train.items()}

        # Initialize and train model
        trainer = MultiOutputTrainer(
            input_size, hidden_size, num_layers, tasks_info, lr=lr
        )
        trainer.train_model(
            train_x=X_train, train_y_dict=y_train,
            val_x=X_val, val_y_dict=y_val,
            epochs=epochs, batch_size=batch_size,
            patience=patience, min_delta=min_delta,
            verbose=verbose
        )

        # Evaluate metrics on test set
        metrics = trainer.evaluate_all_metrics(X_test, y_test)

        # Flatten metrics for this fold
        flat = {'fold': fold}
        for task, m in metrics.items():
            for metric_name, value in m.items():
                if metric_name != 'confusion_matrix':
                    flat[f"{task}_{metric_name}"] = value
        fold_metrics_list.append(flat)

    # Create DataFrame for fold-level metrics
    df_folds = pd.DataFrame(fold_metrics_list).set_index('fold')

    # Compute summary: mean and 95% CI for each metric
    summary = pd.DataFrame(index=['mean', 'ci_lower', 'ci_upper'])
    for col in df_folds.columns:
        data = df_folds[col]
        mean = data.mean()
        sem = data.std(ddof=1) / np.sqrt(folds)
        delta = 1.96 * sem  # 95% CI z-score approximation
        summary[col] = [mean, mean - delta, mean + delta]

    # Display summary
    return df_folds, summary
