"""
2021 Simon Bing, ETHZ, MPI IS
"""
import os
import numpy as np
import logging
from absl import flags, app
# from healthgen.eval.base_evaluator import BaseEvaluator
from evaluation_framework.model_zoo import gru_d_model
import torch
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, confusion_matrix, f1_score
from datetime import datetime
import json
from rich.progress import Progress




class GRUDEvaluator():
    def __init__(self,config):
        
    # def __init__(self, seed, eval_mode,X_test_path = '',y_test_path = '',bagging = False, batch_size=64, hidden_size=64, num_layers=1, dropout=0.1, eval_epochs=100, grud_eval_step=10,eval_early_stop_patience = 20, 
    #              grud_dropout=0.1, grud_lr=0.0005, grud_lr_decay_step=20, grud_l2=0.001, eval_model_path=None, masks_only=False, multi_metric=False, ROC_per_class=None ,out_path = '' ):
        self.input_size = config['input_size']
        self.batch_size = config['batch_size']
        self.hidden_size = config['hidden_size']
        self.num_layers =  config['num_layers']
        self.dropout =  config['dropout']
        self.epochs =  config['epochs']
        self.eval_step = config['eval_step']
        self.dropout = config['dropout']
        self.lr = config['lr']
        self.lr_decay_step = config['lr_decay_step']
        self.l2 = config['l2']
        self.multi_metric = config['multi_metric']
        self.out_path = config['out_path']
        self.eval_early_stop_patience = config['eval_early_stop_patience']
        self.seed = config['seed']
        self.eval_mode = config['eval_mode']

        # Set the seed for torch
        # Set the seed for torch
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # Build model
        self.model = gru_d_model(seed=self.seed, input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 output_size=1, num_layers=self.num_layers,
                                 dropout=self.dropout)
        
    def load(self, saved_eval_model_path):
        self.model.load_state_dict(torch.load(saved_eval_model_path))
        self.model = self.model.to(self.device)

    def train(self, X_train, X_val, y_train, y_val,metadata):
        """
        Trains a new GRU-D evaluator.
        n_smp x 3 x n_channels x len_seq tensor (0: data, 1: mask, 2: deltat)
        
        """
        # Make training output directory
        mode = metadata.split('_')[0]

        # Prepare data for training
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val),torch.Tensor(y_val))
        x_mean = torch.Tensor(X_train[:,0,:,:].mean(axis=(0,2)))

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size,
                                    shuffle=True)

        # Build model
        # input_size = X_train.shape[2]
        self.model = gru_d_model(seed=self.seed, input_size=self.input_size, 
                                 hidden_size=self.hidden_size,
                                 output_size=1, num_layers=self.num_layers,
                                 x_mean=x_mean, dropout=self.dropout)
        logging.info('GRU-D model built!')

        # Prepare training
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
            logging.info('Eval running on GPU')
        else:
            logging.info('Eval running on CPU')

        n_epochs = self.epochs
        criterion = torch.nn.BCELoss()
        

        learning_rate = self.lr
        learning_rate_decay_step = self.lr_decay_step
        l2_penalty = self.l2
        eval_step = self.eval_step
        eval_early_stop_patience = self.eval_early_stop_patience

        # Training loop
        logging.info('Training of GRU-D started!')

        train_num = len(train_dataloader) * self.batch_size
        val_num = len(val_dataloader) * self.batch_size

        train_step = 0
        val_step = 0
        best_val_roc = 0
        best_val_loss = np.inf
        best_epoch = 0
        stop_patience = 0
        patience = eval_early_stop_patience // eval_step
        with Progress() as progress:
            training_task = progress.add_task("[red]Training...", total=self.epochs)
            for epoch in range(n_epochs):
                logging.info(F'Started epoch {epoch} of GRU-D training!')
                if learning_rate_decay_step != 0:
                    if epoch % learning_rate_decay_step == 0: # update learning rate every decay step
                        learning_rate = learning_rate / 2
                        logging.info(F'Updated GRU-D learning rate to {learning_rate}.')

                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                            weight_decay=l2_penalty)

                losses, acc = [], []
                label, pred = [], []
                train_loss = 0

                self.model.train()
                for train_data, train_label in train_dataloader:
                    train_data, train_label = train_data.to(self.device), train_label.to(self.device)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass, take only last RNN output as prediction
                    y_pred = self.model(train_data)[:,-1,:]
                    y_pred = torch.squeeze(y_pred)

                    # Compute loss
                    loss = criterion(y_pred, train_label)

                    # Backward pass
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    acc.append(
                        torch.eq(
                            (torch.sigmoid(y_pred).data > 0.5).float(),
                            train_label)
                    )

                    train_loss += loss.item()
                    train_step = train_step + 1

                # Log (normalized) training loss
                if wandb.run:
                    wandb.log({F'{self.eval_mode}_train_loss': train_loss / len(train_dataloader), 'epoch': epoch})
                progress.update(training_task, advance=1, description=f"Epoch {epoch} - Train Loss: {train_loss / len(train_dataloader)}")

                if epoch % eval_step == 0:
                    logging.info(F'Validating GRU-D at epoch {epoch}!')
                    preds_val = []
                    labels_val = []

                    val_loss = 0

                    self.model.eval()
                    for val_data, val_label in val_dataloader:
                        val_data, val_label = val_data.to(self.device), val_label.to(self.device)

                        # Zero gradients
                        optimizer.zero_grad()

                        # Forward pass, take only last RNN output as prediction
                        y_pred = self.model(val_data)[:,-1,:]
                        y_pred = torch.squeeze(y_pred,dim = 1)

                        # Compute loss
                        loss = criterion(y_pred, val_label)

                        preds_val = np.append(preds_val, y_pred.detach().cpu().numpy())
                        labels_val = np.append(labels_val, val_label.detach().cpu().numpy())

                        val_loss += loss.item()

                        val_step = val_step + 1

                    # Log (normalized) validation loss
                    if wandb.run:
                        wandb.log({F'{self.eval_mode}_val_loss': val_loss/val_num,
                                'epoch': epoch})

                    val_roc = roc_auc_score(labels_val, preds_val)
                    val_prc = average_precision_score(labels_val, preds_val)
                    if val_roc >= best_val_roc:
                        best_model_state_dict = deepcopy(self.model.state_dict())
                        best_val_roc = val_roc
                        best_val_loss = val_loss
                        stop_patience = 0
                        best_epoch = epoch
                        logging.info(F'New best model saved! ROC: {val_roc}')
                    else:
                        stop_patience += 1
                    
                    if stop_patience >= patience:
                        logging.info('Early stopping due to no improvement in validation ROC!')
                        print(f'Early stopping at epoch {epoch} due to no improvement in validation ROC!')
                        break
                    if wandb.run:
                        wandb.log({F'{self.eval_mode}_val_roc': val_roc,
                                F'{self.eval_mode}_val_prc': val_prc,
                                'epoch': epoch})
                
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # best_model_path = os.path.join(self.out_path, 'evaluation_framework', 'downstream', self.eval_mode, f'gru_best_model_{metadata}_{timestamp}.pt')
        best_model_path = os.path.join(self.out_path,f'{mode}_{timestamp}', f'{metadata}_{timestamp}.pt')
        print('Best model saved at:', best_model_path)
        params_path = os.path.join(self.out_path, f'{mode}_{timestamp}', f'{metadata}_params_{timestamp}.json')
        print('Parameters saved at:', params_path)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        logging.info(F'Model saved! {best_model_path}')
        torch.save(best_model_state_dict, best_model_path)
        if wandb.run:
            wandb.config['timestamp'] = timestamp
        self.model.load_state_dict(best_model_state_dict)
        # Save parameters to a JSON file
        params = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'seed': self.seed,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'lr_decay_step': self.lr_decay_step,
            'l2': self.l2,
            'eval_step': self.eval_step,
            'eval_early_stop_patience': self.eval_early_stop_patience,
            'eval_mode': self.eval_mode,
            'out_path': self.out_path,
            'multi_metric': self.multi_metric
        }

        with open(params_path, 'w') as f:
            json.dump(params, f)

        logging.info(F'Parameters saved! {params_path}')
        print('Parameters saved at:', params_path)
        logging.info('GRU-D training successfully finished!')
        print('GRU-D training successfully finished!')
        print('Best model saved at:', best_model_path)
    
    def evaluate(self, X_test, y_test, val =True):
        # # Make output directory
        # eval_base_path = os.path.join(self.out_path, 'eval', 'grud', self.eval_mode)
        # if not os.path.exists(eval_base_path):
        #     os.makedirs(eval_base_path)

        # Evaluation preparation
        test_dataset = TensorDataset(torch.Tensor(X_test),
                                     torch.Tensor(y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     shuffle=True)

        # Compute eval on trained model with test data
        logging.info('Evaluating GRU-D on test data!')
        self.model.eval()
        preds_test = []
        labels_test = []
        for test_data, test_labels in test_dataloader:
            test_data, val_label = test_data.to(
                self.device), test_labels.to(self.device)

            # Forward pass, take only last RNN output as prediction
            y_pred = self.model(test_data)[:, -1, :]
            y_pred = torch.squeeze(y_pred)

            preds_test = np.append(preds_test,
                                   y_pred.detach().cpu().numpy())
            labels_test = np.append(labels_test,
                                    test_labels.detach().cpu().numpy())

        if len(np.unique(labels_test)) == 1:
            logging.warning('Only one class present in y_test. ROC AUC score is not defined in this case.')
            auroc = None
        else:
            auroc = roc_auc_score(labels_test, preds_test)
        if self.multi_metric:
            fpr, tpr, thresholds = roc_curve(labels_test, preds_test)
            tnr = 1 - fpr
            # Find index of closest value to given specificity
            # if val:
            #     fixed_spec = 0.9
            #     thresh_idx = np.argmin((tnr[1:] - fixed_spec)**2) + 1
            #     self.sens_thresh = thresholds[thresh_idx]
            # # Threshold prediction values
            # preds_test_binary_sens = np.where(preds_test >= self.sens_thresh, 1, 0)
            
            if val:
                fixed_sens = 0.9
                # Find the index where sensitivity is closest to the desired fixed sensitivity
                thresh_idx = np.argmin((tpr - fixed_sens)**2)
                self.sens_thresh = thresholds[thresh_idx]
            preds_test_binary_sens = np.where(preds_test >= self.sens_thresh, 1, 0)

            tn, fp, fn, tp = confusion_matrix(labels_test, preds_test_binary_sens, labels = [0,1]).ravel()

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            # ppv = tp / (tp + fp)
            # npv = tn / (tn + fn)
            bacc_sens = (sens + spec) / 2
            f1_sens = f1_score(labels_test, preds_test_binary_sens)

            ### Optimize threshold for other metrics
            J = tpr - fpr
            opt_idx = np.argmax(J[1:]) + 1
            opt_thresh = thresholds[opt_idx]
            # Threshold prediction values
            preds_test_binary_opt = np.where(preds_test >= opt_thresh, 1, 0)
            tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(labels_test, preds_test_binary_opt, labels = [0,1]).ravel()

            sens_opt = tp_opt / (tp_opt + fn_opt)
            spec_opt = tn_opt / (tn_opt + fp_opt)
            bacc_opt = (sens_opt + spec_opt) / 2
            f1_opt = f1_score(labels_test, preds_test_binary_opt)

            metrics = {'auroc': auroc, 'sens': sens, 'spec': spec, 'bacc_sens': bacc_sens,'f1_sens': f1_sens,'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                        'sens_opt':sens_opt, 'spec_opt':spec_opt,'bacc_opt': bacc_opt, 'f1_opt': f1_opt, 'tp_opt': tp, 'tn_opt': tn, 'fp_opt': fp, 'fn_opt': fn}
            return metrics

        else:
            return auroc

    # def bootstrap_evaluate(self, X_test, y_test, n_bootstrap):
    #     bootstrap_metrics = []
    #     for _ in tqdm(range(n_bootstrap)):
    #         # Perform bootstrap sampling with replacement
    #         indices = np.random.choice(len(X_test), len(X_test), replace=True)
    #         X_bootstrap = X_test[indices]
    #         y_bootstrap = y_test[indices]

    #         # Evaluate on bootstrap sample
    #         metrics = self.evaluate(X_bootstrap, y_bootstrap)
    #         bootstrap_metrics.append(metrics)

    #     return bootstrap_metrics

    def bootstrap_evaluate(self, X_test, y_test, n_bootstrap):
        

        metrics_names = ['auroc', 'sens', 'spec', 'bacc_sens','f1_sens','tp', 'tn', 'fp', 'fn','sens_opt','spec_opt','bacc_opt','f1_opt','tp_opt','tn_opt','fp_opt','fn_opt']
        bootstrap_metrics = {metric: [] for metric in metrics_names}
        
        for _ in tqdm(range(n_bootstrap)):
            # Perform bootstrap sampling with replacement
            indices = np.random.choice(len(X_test), len(X_test), replace=True)
            X_bootstrap = X_test[indices]
            y_bootstrap = y_test[indices]

            # Evaluate on bootstrap sample
            metrics = self.evaluate(X_bootstrap, y_bootstrap)
            
            # Append metrics to the corresponding arrays in the dictionary
            for metric in metrics_names:
                bootstrap_metrics[metric].append(metrics[metric])

        return bootstrap_metrics
 



