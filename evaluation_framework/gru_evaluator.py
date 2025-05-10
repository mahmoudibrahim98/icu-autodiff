"""
2021 Simon Bing, ETHZ, MPI IS
"""
import os
import numpy as np
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, confusion_matrix, f1_score
from datetime import datetime
from evaluation_framework.model_zoo import GRUmodel
from tqdm.notebook import trange
from tqdm import tqdm
import json



class GRUEvaluator():
    def __init__(self, config):
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.seed = config['seed']
        self.dropout = config['dropout']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.lr_decay_step = config['lr_decay_step']
        self.l2 = config['l2']
        self.eval_step = config['eval_step']
        self.eval_early_stop_patience = config['eval_early_stop_patience']
        self.eval_mode = config['eval_mode']
        self.out_path = config['out_path']
        self.multi_metric = config['multi_metric']
        self.model = GRUmodel(self.input_size, self.hidden_size, self.num_layers)

    def get_data(self, X_dict=None, y_dict=None,indeces = None):
        """
        Returns:
            X_dict, y_dict: dictionaries of test data and labels
        """
        if X_dict is None and y_dict is None:
            X_dict = np.load(self.X_test_path)
            y_dict = np.load(self.y_test_path)

        X_train, X_val, X_test, X_oracle,\
        y_train, y_val, y_test, y_oracle = self.prepare_data(X_dict, y_dict, indeces = indeces)
        
        return X_train, X_val, X_test, X_oracle, y_train, y_val, y_test, y_oracle
    
    
    @staticmethod
    def slice_array(array, indeces):
        return array[indeces] if indeces is not None else array[:]
    
    
    
    def prepare_subset(self, dict, indeces,type = 'single-X', subset = 'train' ):

        if type == 'single-X':
            name1 = f'X_{subset}'   

        elif type == 'multi-X':
            idx = subset.split('_')[-1]
            sub = subset.split('_')[1]
            name1 = f'X_{sub}_{idx}'

        elif type == 'multi-y':
            idx = subset.split('_')[-1]
            sub = subset.split('_')[1]
            name1 = f'y_{sub}_{idx}'
        elif type == 'single-y':
            name1 = f'y_{subset}'

        def _get_items(array, indices= indeces):
            return array[indices] if indices is not None else array[:]
        
        def _get_items_y(array, indices= indeces):
            if len(array.shape) > 1:
                return array[indices, 0].astype(np.float64) if indices is not None else array[:, 0].astype(np.float64)
            else:
                return array[indices].astype(np.float64) if indices is not None else array[:].astype(np.float64)         
        
        if 'X' in type:
            # return  np.stack((_get_items(dict[name1], indeces), _get_items(dict[name2], indeces), _get_items(dict[name3], indeces)), axis=1)
            temp =  _get_items(dict[name1], indeces)
            return np.transpose(temp, (0, 2, 1))

        else:
            return _get_items_y(dict[name1], indeces)


    def prepare_data(self, X_dict, y_dict, indeces):
        """
        TODO: fill in
        """
        # y_copy = y_dict.copy()

        test_keys = [key for key in X_dict.keys() if key.startswith('X_test_')]

        if self.masks_only:
            print("Masks only eval!")

            X_train = np.stack((np.zeros_like(X_dict['X_train']), X_dict['m_train'], X_dict['delta_t_train']), axis=1)
            X_val = np.stack((np.zeros_like(X_dict['X_val']), X_dict['m_val'], X_dict['delta_t_val']), axis=1)
            if len(test_keys) > 0:
                X_tests = []
                y_tests = []

                for key in X_dict.keys():
                    if key.startswith('X_test_'):
                        idx = key.split('_')[-1]
                        X_tests.append(np.stack((np.zeros_like(X_dict[key]), X_dict[f'm_test_{idx}'], X_dict[f'delta_t_test_{idx}']), axis=1))
                        y_key = f'y_test_{idx}'
                        y_tests.append(y_dict[y_key][:, 0].astype(np.float64) if len(y_dict[y_key].shape) > 1 else y_dict[y_key].astype(np.float64))

            else:
                X_tests = np.stack((np.zeros_like(X_dict['X_test']), X_dict['m_test'], X_dict['delta_t_test']), axis=1)
                X_oracles = np.stack((np.zeros_like(X_dict['X_oracle']), X_dict['m_oracle'], X_dict['delta_t_oracle']), axis=1)

        else:
            # X_train = np.stack((_get_items(X_dict['X_train'], indeces), _get_items(X_dict['m_train'], indeces), _get_items(X_dict['delta_t_train'], indeces)), axis=1)
            X_train = self.prepare_subset(X_dict, indeces,type = 'single-X', subset = 'train')
            # y_train = y_dict['y_train'][:, 0].astype(np.float64) if len(y_dict['y_train'].shape) > 1 else y_dict['y_train'].astype(np.float64)  
            y_train = self.prepare_subset(y_dict, indeces,type = 'single-y', subset = 'train')
            # y_train = _get_items_y(y_dict['y_train'], indeces)
            X_val = self.prepare_subset(X_dict, indeces,type = 'single-X', subset = 'val')
            # X_val = np.stack((_get_items(X_dict['X_val'], indeces), _get_items(X_dict['m_val'], indeces), _get_items(X_dict['delta_t_val'], indeces)), axis=1)
            # y_val = y_dict['y_val'][:, 0].astype(np.float64) if len(y_dict['y_val'].shape) > 1 else y_dict['y_val'].astype(np.float64)
            # y_val = _get_items_y(y_dict['y_val'], indeces)
            y_val = self.prepare_subset(y_dict, indeces,type = 'single-y', subset = 'val')
            if len(test_keys) > 0:
                X_tests = []
                y_tests = []

                for key in X_dict.keys():
                    if key.startswith('X_test_'):
                        idx = key.split('_')[-1]
                        y_key = f'y_test_{idx}'
                        # X_tests.append(np.stack((X_dict[key], X_dict[f'm_test_{idx}'], X_dict[f'delta_t_test_{idx}']), axis=1))
                        X_tests.append(self.prepare_subset(X_dict, indeces,type = 'multi-X', subset = key))
                        # X_tests.append(np.stack((_get_items(X_dict[key], indeces), _get_items(X_dict[f'm_test_{idx}'], indeces), _get_items(X_dict[f'delta_t_test_{idx}'], indeces)), axis=1))
                        # y_tests.append(y_dict[y_key][:, 0].astype(np.float64) if len(y_dict[y_key].shape) > 1 else y_dict[y_key].astype(np.float64))
                        # y_tests.append(_get_items_y(y_dict[y_key], indeces))
                        y_tests.append(self.prepare_subset(y_dict, indeces,type = 'multi-y', subset = key))
                if self.eval_mode == 'real':
                    X_oracles = []
                    y_oracles = []
                    for key in X_dict.keys():
                        if key.startswith('X_oracle_'):
                            idx = key.split('_')[-1]
                            y_key = f'y_oracle_{idx}'
                            # X_oracles.append(np.stack((X_dict[key], X_dict[f'm_oracle_{idx}'], X_dict[f'delta_t_oracle_{idx}']), axis=1))
                            # X_oracles.append(np.stack((_get_items(X_dict[key], indeces), _get_items(X_dict[f'm_oracle_{idx}'], indeces), _get_items(X_dict[f'delta_t_oracle_{idx}'], indeces)), axis=1))
                            X_oracles.append(self.prepare_subset(X_dict, indeces,type = 'multi-X', subset = key))
                            # y_oracles.append(y_dict[y_key][:, 0].astype(np.float64) if len(y_dict[y_key].shape) > 1 else y_dict[y_key].astype(np.float64))
                            # y_oracles.append(_get_items_y(y_dict[y_key], indeces))
                            y_oracles.append(self.prepare_subset(y_dict, indeces,type = 'multi-y', subset = key))
                else:
                    X_oracles = None
                    y_oracles = None
            else:
                # X_tests = np.stack((X_dict['X_test'], X_dict['m_test'], X_dict['delta_t_test']), axis=1)
                # X_tests = np.stack((_get_items(X_dict['X_test'], indeces), _get_items(X_dict['m_test'], indeces), _get_items(X_dict['delta_t_test'], indeces)), axis=1)
                X_tests = self.prepare_subset(X_dict, indeces,type = 'single-X', subset = 'test')
                # y_tests = y_dict['y_test'][:, 0].astype(np.float64) if len(y_dict['y_test'].shape) > 1 else y_dict['y_test'].astype(np.float64)
                # y_tests = _get_items_y(y_dict['y_test'], indeces)   
                y_tests = self.prepare_subset(y_dict, indeces,type = 'single-y', subset = 'test')
                if self.eval_mode == 'real':
                    # X_oracles = np.stack((X_dict['X_oracle'], X_dict['m_oracle'], X_dict['delta_t_oracle']), axis=1)
                    # X_oracles = np.stack((_get_items(X_dict['X_oracle'], indeces), _get_items(X_dict['m_oracle'], indeces), _get_items(X_dict['delta_t_oracle'], indeces)), axis=1)
                    X_oracles = self.prepare_subset(X_dict, indeces,type = 'single-X', subset = 'oracle')
                    # y_oracles = y_dict['y_oracle'][:, 0].astype(np.float64) if len(y_dict['y_oracle'].shape) > 1 else y_dict['y_oracle'].astype(np.float64)
                    # y_oracles = _get_items_y(y_dict['y_oracle'], indeces)
                    y_oracles = self.prepare_subset(y_dict, indeces,type = 'single-y', subset = 'oracle')
                else:
                    X_oracles = None
                    y_oracles = None
        # Need input size to build model
        self.input_size = X_train.shape[2]
        return  X_train, X_val, X_tests, X_oracles, y_train, y_val , y_tests, y_oracles

    def load(self, saved_eval_model_path):
        self.model.load_state_dict(torch.load(saved_eval_model_path))
        self.model = self.model.to(self.device)
        
    def standardize(self, X, train = True):
        from sklearn.preprocessing import StandardScaler
        # Reshape the data then standardize it
        reshaped_X = X.reshape(-1, X.shape[2])
        if train:
            scaler = StandardScaler()
            scaler = scaler.fit(reshaped_X)
            self.scaler = scaler
            normalized_X = scaler.transform(reshaped_X).reshape(X.shape)
            return normalized_X
        else:
            scaler = self.scaler
            normalized_X = scaler.transform(reshaped_X).reshape(X.shape)
            return normalized_X



    def train(self, X_train, X_val, y_train, y_val, metadata, preprocess_standardize = True, verbose = True):
        # train_base_path = os.path.join(self.out_path, 'evaluation_framework', 'downstream', self.eval_mode, 'training')
        # if not os.path.exists(train_base_path):
        #     os.makedirs(train_base_path)
        if preprocess_standardize:
            self.preprocess_standardize = True
            X_train = self.standardize(X_train,train= True)
            X_val = self.standardize(X_val,train= False)
        # mode = metadata.split('_')[0]
        mode = metadata
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = GRUmodel(self.input_size, self.hidden_size, self.num_layers)
        self.model = self.model.to(self.device)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
            logging.info('Eval running on GPU')
        else:
            logging.info('Eval running on CPU')

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)

        best_val_roc = 0
        best_val_loss = np.inf
        best_epoch = 0
        stop_patience = 0
        patience = self.eval_early_stop_patience // self.eval_step
        # tqdm_epoch = tqdm(range(self.epochs))
        # Initialize tqdm for epochs
        if verbose:
            tqdm_epoch = tqdm(range(self.epochs), desc="Training", leave=True)
        else:
            tqdm_epoch = range(self.epochs)

        for epoch in tqdm_epoch:
            logging.info(F'Started epoch {epoch} of GRU training!')
            
            # Adjust learning rate if needed
            if self.lr_decay_step != 0 and epoch % self.lr_decay_step == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2
                logging.info(F'Updated GRU learning rate to {optimizer.param_groups[0]["lr"]}.')
 
            self.model.train()
            train_loss = 0

            # Batch-wise tqdm for training data
            batch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for train_data, train_label in batch_bar:
                train_data, train_label = train_data.to(self.device), train_label.to(self.device)
                optimizer.zero_grad()
                y_hat_logit, y_hat = self.model(train_data)
                y_hat = torch.squeeze(y_hat)
                
                # Fix: Ensure compatible shapes for loss calculation
                if y_hat.dim() == 0 and train_label.dim() > 0:
                    y_hat = y_hat.unsqueeze(0)
                elif train_label.dim() == 0 and y_hat.dim() > 0:
                    train_label = train_label.unsqueeze(0)
                    
                loss = criterion(y_hat, train_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Update batch progress bar with current batch loss
                batch_bar.set_postfix(loss=loss.item())

            batch_bar.close()  # Close wha progress bar

            # Log train loss for the epoch
            avg_train_loss = train_loss / len(train_dataloader)
            if wandb.run:
                wandb.log({F'{self.eval_mode}_train_loss': avg_train_loss, 'epoch': epoch})

            if verbose: 
                tqdm_epoch.set_postfix(train_loss=avg_train_loss)  # Update epoch progress bar description

            # Validation every eval_step epochs
            if epoch % self.eval_step == 0:
                logging.info(F'Validating GRU at epoch {epoch}!')
                val_loss = 0
                preds_val = []
                labels_val = []

                self.model.eval()
                for val_data, val_label in val_dataloader:
                    val_data, val_label = val_data.to(self.device), val_label.to(self.device)
                    with torch.no_grad():
                        y_hat_logit, y_hat = self.model(val_data)
                        y_hat = torch.squeeze(y_hat)
                        # Fix: Ensure compatible shapes for loss calculation
                        if y_hat.dim() == 0 and val_label.dim() > 0:
                            y_hat = y_hat.unsqueeze(0)
                        elif val_label.dim() == 0 and y_hat.dim() > 0:
                            val_label = val_label.unsqueeze(0)
                        loss = criterion(y_hat, val_label)
                        val_loss += loss.item()
                        preds_val.extend(y_hat.detach().cpu().numpy())
                        labels_val.extend(val_label.detach().cpu().numpy())

                val_roc = roc_auc_score(labels_val, preds_val)
                val_prc = average_precision_score(labels_val, preds_val)

                # Check for the best model and early stopping
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
                    wandb.log({
                        F'{self.eval_mode}_val_roc': val_roc,
                        F'{self.eval_mode}_val_prc': val_prc,
                        'epoch': epoch
                    })
        if verbose:
            tqdm_epoch.close()  # Close epoch progress bar
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # best_model_path = os.path.join(self.out_path, 'evaluation_framework', 'downstream', self.eval_mode, f'gru_best_model_{metadata}_{timestamp}.pt')
        best_model_path = os.path.join(self.out_path,f'{mode}_{timestamp}', f'{metadata}_{timestamp}.pt')
        # print('Best model saved at:', best_model_path)
        params_path = os.path.join(self.out_path, f'{mode}_{timestamp}', f'{metadata}_params_{timestamp}.json')
        # print('Parameters saved at:', params_path)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        model_path = os.path.join(self.out_path,f'{mode}_{timestamp}')

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
        # print('Parameters saved at:', params_path)
        logging.info('GRU training successfully finished!')
        # print('GRU training successfully finished!')
        # print('Best model saved at:', best_model_path)
        return model_path
    def predict_test(self, X_test, y_test):
        # eval_base_path = os.path.join(self.out_path, 'eval', 'grud', self.eval_mode)
        # if not os.path.exists(eval_base_path):
        #     os.makedirs(eval_base_path)

        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        logging.info('Evaluating GRU on test data!')
        self.model.eval()
        preds_test = []
        labels_test = []

        for test_data, test_labels in test_dataloader:
            test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)
            y_hat_logit, y_hat = self.model(test_data)
            # y_hat = torch.squeeze(y_hat)
            y_hat = y_hat.view(-1)
            preds_test.extend(y_hat.detach().cpu().numpy())
            labels_test.extend(test_labels.detach().cpu().numpy())

        return preds_test, labels_test
        
    def evaluate(self, X_test, y_test, val=True, verbose =False):
        if self.preprocess_standardize:
            X_test = self.standardize(X_test,train= False)

        # eval_base_path = os.path.join(self.out_path, 'eval', 'grud', self.eval_mode)
        # if not os.path.exists(eval_base_path):
        #     os.makedirs(eval_base_path)

        # test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        # test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        # logging.info('Evaluating GRU on test data!')
        # self.model.eval()
        # preds_test = []
        # labels_test = []

        # for test_data, test_labels in test_dataloader:
        #     test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)
        #     y_hat_logit, y_hat = self.model(test_data)
        #     # y_hat = torch.squeeze(y_hat)
        #     y_hat = y_hat.view(-1)
        #     preds_test.extend(y_hat.detach().cpu().numpy())
        #     labels_test.extend(test_labels.detach().cpu().numpy())
        preds_test, labels_test = self.predict_test(X_test, y_test)
        
        if len(np.unique(labels_test)) == 1:
            if verbose:
                logging.warning('Only one class present in y_test. ROC AUC score is not defined in this case.')
            auroc = None
            auprc = None
        else:
            auroc = roc_auc_score(labels_test, preds_test)
            auprc = average_precision_score(labels_test, preds_test)
        if self.multi_metric:
            fpr, tpr, thresholds = roc_curve(labels_test, preds_test)
            tnr = 1 - fpr
            if val:
                fixed_sens = 0.9
                thresh_idx = np.argmin((tpr - fixed_sens)**2)
                self.sens_thresh = thresholds[thresh_idx]
            preds_test_binary_sens = np.where(preds_test >= self.sens_thresh, 1, 0)

            tn, fp, fn, tp = confusion_matrix(labels_test, preds_test_binary_sens, labels = [0,1]).ravel()
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            bacc_sens = (sens + spec) / 2
            f1_sens = f1_score(labels_test, preds_test_binary_sens)

            J = tpr - fpr
            opt_idx = np.argmax(J[1:]) + 1
            opt_thresh = thresholds[opt_idx]
            preds_test_binary_opt = np.where(preds_test >= opt_thresh, 1, 0)
            tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(labels_test, preds_test_binary_opt, labels = [0,1]).ravel()
            sens_opt = tp_opt / (tp_opt + fn_opt)
            spec_opt = tn_opt / (tn_opt + fp_opt)
            bacc_opt = (sens_opt + spec_opt) / 2
            f1_opt = f1_score(labels_test, preds_test_binary_opt)

            metrics = {
                'auroc': auroc,
                'auprc': auprc,
                'sens': sens,
                'spec': spec,
                'bacc_sens': bacc_sens,
                'f1_sens': f1_sens,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'sens_opt': sens_opt,
                'spec_opt': spec_opt,
                'bacc_opt': bacc_opt,
                'f1_opt': f1_opt,
                'tp_opt': tp_opt,
                'tn_opt': tn_opt,
                'fp_opt': fp_opt,
                'fn_opt': fn_opt
            }
            return metrics
        else:
            return auroc
        
        
    def bootstrap_evaluate(self, X_test, y_test, n_bootstrap, verbose = False):
        

        metrics_names = ['auroc', 'sens', 'spec', 'bacc_sens','f1_sens','tp', 'tn', 'fp', 'fn','sens_opt','spec_opt','bacc_opt','f1_opt','tp_opt','tn_opt','fp_opt','fn_opt']
        bootstrap_metrics = {metric: [] for metric in metrics_names}
        
        if verbose:
            for _ in tqdm(range(n_bootstrap)):
                # Perform bootstrap sampling with replacement
                indices = np.random.choice(len(X_test), len(X_test), replace=True)
                X_bootstrap = X_test[indices]
                y_bootstrap = y_test[indices]

                # Evaluate on bootstrap sample
                metrics = self.evaluate(X_bootstrap, y_bootstrap,verbose)
                
                # Append metrics to the corresponding arrays in the dictionary
                for metric in metrics_names:
                    bootstrap_metrics[metric].append(metrics[metric])
        else:
            for _ in range(n_bootstrap):
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
 

