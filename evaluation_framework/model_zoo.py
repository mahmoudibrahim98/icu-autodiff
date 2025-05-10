import abc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import trange
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score,roc_auc_score, average_precision_score, confusion_matrix, f1_score, roc_curve
from datetime import datetime
import os
import logging
import wandb
from copy import deepcopy
from tqdm import tqdm
from evaluation_framework.metric_utils import extract_time
from sklearn.metrics import accuracy_score,mean_absolute_error
import numpy as np
import pandas as pd
import os
import math
import warnings
import itertools
import numbers
import torch.utils.data as utils
from evaluation_framework.gru_d_layer import GRUD_cell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class gru_d_model(nn.Module):
    def __init__(self, seed, input_size, hidden_size, output_size, num_layers=1,
                 x_mean=0.0, \
                 bias=True, batch_first=False, bidirectional=False,
                 dropout_type='mloss', dropout=0):
        super(gru_d_model, self).__init__()

        torch.manual_seed(seed)

        self.gru_d = GRUD_cell(seed=seed, input_size=input_size, hidden_size=hidden_size,
                               output_size=output_size,
                               dropout=dropout, dropout_type=dropout_type,
                               x_mean=x_mean)
        self.hidden_to_output = torch.nn.Linear(hidden_size, output_size,
                                                bias=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if self.num_layers > 1:
            # (batch, seq, feature)
            self.gru_layers = torch.nn.GRU(input_size=hidden_size,
                                           hidden_size=hidden_size,
                                           batch_first=True,
                                           num_layers=self.num_layers - 1,
                                           dropout=dropout)

    def initialize_hidden(self, batch_size):
        device = next(self.parameters()).device
        # The hidden state at the start are all zeros
        return torch.zeros(self.num_layers - 1, batch_size, self.hidden_size,
                           device=device)

    def forward(self, input):

        # pass through GRU-D
        output, hidden = self.gru_d(input)
        # print(self.gru_d.return_hidden)
        # output = self.gru_d(input)
        # print(output.size())

        # batch_size, n_hidden, n_timesteps

        if self.num_layers > 1:
            # TODO remove init hidden, not necessary, auto init works fine
            init_hidden = self.initialize_hidden(hidden.size()[0])

            output, hidden = self.gru_layers(hidden)  # , init_hidden)

            output = self.hidden_to_output(output)
            output = torch.sigmoid(output)

        # print("final output size passed as model result")
        # print(output.size())
        return output



class BaseDownstreamEvaluator(abc.ABC):
    @abc.abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
def batch_generator(data, batch_size):
    idx = np.random.permutation(len(data))
    for start_idx in range(0, len(data) - batch_size + 1, batch_size):
        excerpt = idx[start_idx:start_idx + batch_size]
        yield data[excerpt]

       
class GRUmodel(nn.Module):
    
    # Input Shape: (batch_size, sequence_length, input_size)
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.RNN = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, d_last_states = self.RNN(x)
        y_hat_logit = self.fc(d_last_states[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat

class GRUDiscriminator(BaseDownstreamEvaluator):
    def __init__(self, input_size, hidden_size, num_layers):
        self.discriminator = GRUmodel(input_size, hidden_size, num_layers)
        self.discriminator = torch.nn.DataParallel(self.discriminator).to(device)
        self.optimizer = optim.Adam(self.discriminator.parameters())

    def train_model(self, train_x, train_x_hat,iterations =  1000, val_x=None, val_y=None, metadata=None , verbose = False):
        batch_size = 128
        # tqdm_epoch = trange(iterations)
        if verbose:
            tqdm_epoch = tqdm(range(iterations))
        else:
            tqdm_epoch = range(iterations)

        losses = []

        for itt in tqdm_epoch:
            if isinstance(train_x, np.ndarray):
                train_x = torch.tensor(train_x, dtype=torch.float32)
            if isinstance(train_x_hat, np.ndarray):
                train_x_hat = torch.tensor(train_x_hat, dtype=torch.float32)
            X_mb = next(batch_generator(train_x, batch_size)).to(device)
            X_hat_mb = next(batch_generator(train_x_hat, batch_size)).to(device)

            self.optimizer.zero_grad()

            y_logit_real, _ = self.discriminator(X_mb)
            y_logit_fake, _ = self.discriminator(X_hat_mb)

            d_loss_real = nn.BCEWithLogitsLoss()(y_logit_real, torch.ones_like(y_logit_real))
            d_loss_fake = nn.BCEWithLogitsLoss()(y_logit_fake, torch.zeros_like(y_logit_fake))
            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            self.optimizer.step()
            
            losses.append(d_loss.item())
            if verbose:
                tqdm_epoch.set_description(f'Average Loss: {d_loss.item():.5f}')

    def evaluate(self, test_x, test_x_hat):
        model_device = next(self.discriminator.module.parameters()).device
        if isinstance(test_x, np.ndarray):
            test_x = torch.tensor(test_x, dtype=torch.float32)
        if isinstance(test_x_hat, np.ndarray):
            test_x_hat = torch.tensor(test_x_hat, dtype=torch.float32)
        test_x = test_x.to(model_device)
        test_x_hat = test_x_hat.to(model_device)
        with torch.no_grad():
            _, y_pred_real_curr = self.discriminator(test_x)
            _, y_pred_fake_curr = self.discriminator(test_x_hat)

        y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr.detach().cpu().numpy(), y_pred_fake_curr.detach().cpu().numpy()), axis=0))
        y_label_final = np.concatenate((np.ones(len(y_pred_real_curr)), np.zeros(len(y_pred_fake_curr))), axis=0)

        acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
        discriminative_score = np.abs(0.5 - acc)

        return discriminative_score
 

class GRUpredmodel(nn.Module):
    """Simple predictor network using GRU."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUpredmodel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Pack padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, t.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # GRU forward
        packed_output, _ = self.gru(packed_x)
        
        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        # Fully connected layer
        logits = self.fc(output)
        predictions = self.sigmoid(logits)
        
        return predictions
    
           
class GRUPredictor(BaseDownstreamEvaluator):
    def __init__(self, dim, hidden_dim):
        self._predictor = GRUpredmodel(input_dim=dim-1, hidden_dim=hidden_dim).to(device)

    def train_model(self, ori_data, generated_data, iterations, verbose):
        if verbose:
            tqdm_epoch = tqdm(range(iterations))
        else:
            tqdm_epoch = range(iterations)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        no, seq_len, dim = np.asarray(ori_data).shape

        # Set maximum sequence length and each sequence length
        ori_time, ori_max_seq_len = extract_time(ori_data)
        generated_time, generated_max_seq_len = extract_time(generated_data)
        max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

        # Network parameters
        batch_size = 128
        # Initialize the model, loss function, and optimizer
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self._predictor.parameters())

        # Training using Synthetic dataset
        self._predictor.train()
        for _ in tqdm_epoch:
            # Set mini-batch
            idx = np.random.permutation(len(generated_data))
            train_idx = idx[:batch_size]

            X_mb = [generated_data[i][:-1, :(dim-1)] for i in train_idx]
            T_mb = [generated_time[i] - 1 for i in train_idx]
            Y_mb = [generated_data[i][1:, (dim-1)].reshape(-1, 1) for i in train_idx]

            # Pad sequences
            X_mb = nn.utils.rnn.pad_sequence([torch.FloatTensor(x) for x in X_mb], batch_first=True).to(device)
            Y_mb = nn.utils.rnn.pad_sequence([torch.FloatTensor(y) for y in Y_mb], batch_first=True).to(device)
            T_mb = torch.LongTensor(T_mb).to(device)

            # Forward pass
            optimizer.zero_grad()
            y_pred = self._predictor(X_mb, T_mb)

            # Compute loss only on non-padded elements
            mask = torch.arange(Y_mb.size(1))[None, :].to(device) < T_mb[:, None]
            loss = criterion(y_pred[mask], Y_mb[mask])

            # Backward pass
            loss.backward()
            optimizer.step()

    def evaluate(self,ori_data, ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        no, seq_len, dim = np.asarray(ori_data).shape

        # Set maximum sequence length and each sequence length
        ori_time, ori_max_seq_len = extract_time(ori_data)

        # Test the trained model on the original data
        self._predictor.eval()
        with torch.no_grad():
            idx = np.random.permutation(len(ori_data))
            test_idx = idx[:no]

            # Prepare test data
            X_mb = [ori_data[i][:-1, :(dim-1)] for i in test_idx]
            T_mb = [ori_time[i] - 1 for i in test_idx]
            Y_mb = [ori_data[i][1:, (dim-1)].reshape(-1, 1) for i in test_idx]

            # Pad sequences
            X_mb = nn.utils.rnn.pad_sequence([torch.FloatTensor(x) for x in X_mb], batch_first=True).to(device)
            T_mb = torch.LongTensor(T_mb).to(device)

            # Get predictions
            pred_Y = self._predictor(X_mb, T_mb).cpu().numpy()

            # Compute MAE for each sequence and average
            MAE_temp = 0
            for i in range(no):
                # Only compute MAE up to the actual sequence length
                MAE_temp += mean_absolute_error(Y_mb[i][:T_mb[i].item()],
                                                pred_Y[i][:T_mb[i].item()])

            predictive_score = MAE_temp / no

        return predictive_score



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_hat = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        return y_hat


class LSTMCondtionalEvaluator(BaseDownstreamEvaluator):
    # LSTM is consdiered for conditional evaluator instead of the GRU beacuse LSTM is 
    # more suitable for multi-label classification
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        self.model = torch.nn.DataParallel(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters())

    def train_model(self, train_x, train_y, val_x=None, val_y=None, metadata=None, iterations=1000):
        batch_size = 128
        criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
        tqdm_epoch = tqdm.tqdm(range(iterations))
        losses = []

        for itt in tqdm_epoch:
            self.model.train()
            self.optimizer.zero_grad()
            
            idx = np.random.choice(len(train_x), batch_size, replace=False)
            X_mb = torch.FloatTensor(train_x[idx]).to(device)
            y_mb = torch.FloatTensor(train_y[idx]).to(device)
            
            y_hat = self.model(X_mb)
            loss = criterion(y_hat, y_mb)
            
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            tqdm_epoch.set_description(f'Average Loss: {loss.item():.5f}')
            
            if val_x is not None and val_y is not None:
                self.model.eval()
                with torch.no_grad():
                    val_x_mb = torch.FloatTensor(val_x).to(device)
                    val_y_mb = torch.FloatTensor(val_y).to(device)
                    val_y_hat = self.model(val_x_mb)
                    val_loss = criterion(val_y_hat, val_y_mb)
                    tqdm_epoch.set_postfix({'Validation Loss': val_loss.item()})

    def evaluate(self, test_x, test_y):
        self.model.eval()
        with torch.no_grad():
            test_x = torch.FloatTensor(test_x).to(device)
            test_y = torch.FloatTensor(test_y).to(device)
            y_hat = self.model(test_x)
            y_pred = torch.sigmoid(y_hat).cpu().numpy()  # Get probabilities

        roc_auc = roc_auc_score(test_y.cpu().numpy(), y_pred, average='micro')  # Calculate ROC AUC

        return roc_auc


