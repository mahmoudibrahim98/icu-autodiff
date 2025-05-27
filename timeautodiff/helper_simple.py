import sys
# TODO

from datetime import datetime, timedelta
import wandb
import pandas as pd
import seaborn as sns
import ast
import logging
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())


from absl import flags

import timeautodiff.processing_simple as processing
import timeautodiff.timeautodiff_v4_efficient_simple as timeautodiff
import evaluation_framework.metric_utils as metric_utils
import evaluation_framework.gru_evaluator as gru_evaluator



##################################################################
####          Append new parameters to metadata               ####
##################################################################

def append_new_params_to_metadata(output_dir, new_params):
    # Path to the metadata JSON file
    metadata_path = os.path.join(output_dir, 'metadata.json')

    # Read the existing JSON file
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Append the new parameters
    metadata.update(new_params)

    # Write the updated metadata back to the JSON file
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("New parameters appended successfully.")

def read_data_from_json(output_dir):
    metadata_path = os.path.join(output_dir, 'metadata.json')

    with open(metadata_path, 'r') as f:
        data = json.load(f)
    return data



##################################################################
####          Independent Model Training                     ####
##################################################################

def train_independent_models(down_train_X, down_val_X, down_train_y, down_val_y,num_models,output_dir, verbose = False):
    print('Independent Model Training')
    train_scores = []
    val_scores = []
    downstream_models = []

    for i in range(num_models):  # careful: since the train/val split is fixed, there is no need to train 10 times
        dim = down_train_X.shape[2]
        evaluation = 'gru'
        
        gru_params = {
            'input_size': dim,  
            'batch_size': 64,
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'epochs': 100,
            'eval_step': 10,
            'lr': 0.0005,
            'lr_decay_step': 20,
            'l2': 0.001,
            'multi_metric': True,
            'evaluation': 'gru',
            'eval_early_stop_patience': 20,
            'out_path': os.path.join(output_dir, 'gru/trts/'),
            'eval_mode': 'trts',
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'seed': 0
        }

        downstream_evaluator = gru_evaluator.GRUEvaluator(gru_params)
        model_path = downstream_evaluator.train(down_train_X, down_val_X, down_train_y, down_val_y, 'intersectional')
        
        train_score = downstream_evaluator.evaluate(down_train_X, down_train_y)['auroc']
        val_score = downstream_evaluator.evaluate(down_val_X, down_val_y)['auroc']
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        downstream_models.append(downstream_evaluator)
        if verbose:
            print(f"Model {i}, Train Score: {train_score}, Val Score: {val_score}")
    if verbose:
        metric_utils.display_scores(train_scores)
        metric_utils.display_scores(val_scores)
    return downstream_models

def load_independent_models(down_train_X, down_val_X, down_train_y, down_val_y,num_models,output_dir, verbose = False):
    print('Independent Model Loading')
    train_scores = []
    val_scores = []
    downstream_models = []

    for i,model_Stamp in enumerate(os.listdir(os.path.join(output_dir, 'gru/trts/'))[-num_models:]):
        dim = down_train_X.shape[2]
        evaluation = 'gru'
        
        gru_params = {
            'input_size': dim,  
            'batch_size': 64,
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'epochs': 100,
            'eval_step': 10,
            'lr': 0.0005,
            'lr_decay_step': 20,
            'l2': 0.001,
            'multi_metric': True,
            'evaluation': 'gru',
            'eval_early_stop_patience': 20,
            'out_path': os.path.join(output_dir, 'gru/trts/'),
            'eval_mode': 'trts',
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'seed': 0
        }

        downstream_evaluator = gru_evaluator.GRUEvaluator(gru_params)
        downstream_evaluator.load(os.path.join(output_dir, 'gru/trts/',model_Stamp,model_Stamp+'.pt'))
        downstream_evaluator.preprocess_standardize = True
        
        _ = downstream_evaluator.standardize(down_train_X, train = True)
        train_score = downstream_evaluator.evaluate(down_train_X, down_train_y)['auroc']
        val_score = downstream_evaluator.evaluate(down_val_X, down_val_y)['auroc']
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        downstream_models.append(downstream_evaluator)
        if verbose:
            print(f"Model {i}, Train Score: {train_score}, Val Score: {val_score}")
    if verbose:
        metric_utils.display_scores(train_scores)
        metric_utils.display_scores(val_scores)
    return downstream_models

##################################################################
####              Load AE and Diffusion Model                ####
##################################################################

def load_models_only(latest_timestamp, task_name, data_name,checkpoint_vae = None,checkpoint_diff = None, gen_model = 'TimeAutoDiff'):

    latest_dir = f'outputs/{task_name}/{data_name}/{gen_model}/{latest_timestamp}'
    output_dir = latest_dir


    # Load Auto-encoder
    if checkpoint_vae is None:
        filepath = os.path.join(latest_dir, 'autoencoder')
        latent_features = torch.load(os.path.join(latest_dir, 'latent_features.pt'))
        loaded_autoencoder = timeautodiff.DeapStack.load_model(filepath + '.pt')
    else:
        filepath = os.path.join(latest_dir,'autoencoder_checkpoints', 'model_epoch_{}.pth'.format(checkpoint_vae))   
        latent_features = torch.load(os.path.join(latest_dir,'autoencoder_checkpoints', 'latent_features_epoch_{}.pt'.format(checkpoint_vae)))
        loaded_autoencoder = timeautodiff.DeapStack.load_model(filepath + '.pt')
    # Load saved latent features
    print("Latent features loaded successfully.")
    print("Latent features shape:", latent_features.shape)
    
    # Load Diffusion Model
    if checkpoint_diff is None:
        filepath = os.path.join(latest_dir, 'diffusion')
        loaded_diffusion = timeautodiff.BiRNN_score.load_model(filepath + '.pt')
    else:
        filepath = os.path.join(latest_dir,'diff_checkpoints', 'model_epoch_{}.pth'.format(checkpoint_diff))
        loaded_diffusion = timeautodiff.BiRNN_score.load_model(filepath + '.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    return {
        'ae': loaded_autoencoder,
        'latent_features': latent_features,
        'diff': loaded_diffusion,
        'device': device,
        'output_dir': output_dir,

    }
    
##################################################################
####                  GENERATE SYNTHETIC DATA                 ####
##################################################################
def restore_original_order_tensor_argsort(shuffled_tensor, col_order):
    """Alternative implementation using argsort for tensors"""
    # Convert col_order to a tensor if it's not already
    if not isinstance(col_order, torch.Tensor):
        col_order = torch.tensor(col_order, device=shuffled_tensor.device)
    
    # Get the inverse permutation using argsort
    inverse_order = torch.argsort(col_order)
    
    # Apply the inverse permutation
    original_tensor = shuffled_tensor[:, :, inverse_order]  # For 3D tensor
    
    return original_tensor

def generate_synthetic_data_simple(models, cond, time_info, numerical_processing = 'normalize', unprocess = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = models['ae'].to(device)
    diff = models['diff'].to(device)

    Batch_size, Seq_len, _ = cond.shape
    lat_dim = models['ae'].lat_dim
    
    # Create time grid and ensure correct shape
    t_grid = torch.linspace(0, 1, Seq_len).view(1, -1, 1).to(device).repeat(Batch_size, 1, 1)
    
    # Generate samples
    samples = timeautodiff.sample(t_grid, Batch_size, Seq_len, lat_dim, diff, cond, time_info)

    # Apply decoder to generated latent vector
    gen_output = ae.decoder(samples)
    
    datatype_info = {
        'n_bins': models['ae'].n_bins,
        'n_cats': models['ae'].n_cats,
        'n_nums': models['ae'].n_nums,
        'cards': models['ae'].cards
    }
    col_order = models['ae'].column_order
    synth_data = processing.convert_to_tensor(gen_output, Batch_size, Seq_len, datatype_info) # binary -> categorical -> numerical  
    synth_data = restore_original_order_tensor_argsort(synth_data, col_order) # restore the original order of the data
    
    ### if we dont want to unprocess ( unnormalize) the data, 
    # no need to call the conver_to_table, as the data is already in the correct format
    return synth_data

def generate_synthetic_data_in_batches(models, cond, time_info, batch_size = 10000):


    # Initialize lists to hold the generated synthetic data
    synth_data_batches = []

    # Get the total number of samples
    num_samples = cond.size(0)

    # Iterate over the data in batches
    for start in range(0, num_samples, batch_size):
        
        end = min(start + batch_size, num_samples)  # Ensure we don't go out of bounds
        cond_batch = cond[start:end]
        time_info_batch = time_info[start:end]
        

        _synth_data = generate_synthetic_data_simple(models, cond_batch, time_info_batch)


        # Append the results to the lists
        synth_data_batches.append(_synth_data)

    # Concatenate the results from all batches 
    synth_data = torch.cat(synth_data_batches, dim=0)
    return synth_data
def calculate_missing_sequences(data_array, feature_names):
    """
    Calculate the percentage of missing values for each feature across all available points.
    """
    # Initialize arrays to store the count of missing values for each feature
    missing_values_counts = np.zeros(data_array.shape[1])
    total_values_counts = np.zeros(data_array.shape[1])
    missing_values = {}
    # Iterate over each feature
    for feature_idx in range(data_array.shape[1]):
        # Count the number of missing values for this feature
        missing_values_counts[feature_idx] = np.isnan(data_array[:, feature_idx, :]).sum()
        # Count the total number of values for this feature
        total_values_counts[feature_idx] = data_array[:, feature_idx, :].size

    # Calculate the percentage of missing values for each feature
    missing_percentage = (missing_values_counts / total_values_counts) * 100

    # Create a DataFrame to display the results
    missing_values_df = pd.DataFrame({
        'Feature': feature_names,
        'Missing Values Count': missing_values_counts,
        'Total Values Count': total_values_counts,
        'Missing Percentage (%)': missing_percentage
    })
    missing_values[feature_idx] = missing_percentage
    missing_values_df = missing_values_df.sort_values(by='Missing Percentage (%)', ascending=True)

    return missing_values_df, missing_values



def filter_patients_with_null_values(X_original,c, feature_names, specified_features):
    # Get the indices of the specified features
    specified_feature_indices = [np.where(feature_names == feature)[0][0] for feature in specified_features]

    # Initialize a boolean array to keep track of patients to keep
    patients_to_keep = np.ones(X_original.shape[0], dtype=bool)

    # Iterate over each specified feature index
    for feature_idx in specified_feature_indices:
        # Check if somes values in the sequence for a patient are NaN for the specified feature
        missing_sequences = np.any(np.isnan(X_original[:, feature_idx, :]), axis=1)
        # Update the patients_to_keep array
        patients_to_keep &= ~missing_sequences
    if len(c) > 0:
        for cond in range(c.shape[1]):
            missing_sequences = np.isnan(c[:, cond])
            patients_to_keep &= ~missing_sequences

        

    print(f"Original number of patients: {X_original.shape[0]}")
    print(f"Number of patients after filtering: {sum(patients_to_keep)}")

    return patients_to_keep


# Function to print CUDA memory statistics
def print_cuda_memory_stats():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = reserved_memory - allocated_memory

        print(f"Total CUDA memory: {total_memory / (1024 ** 3):.2f} GiB")
        print(f"Reserved CUDA memory: {reserved_memory / (1024 ** 3):.2f} GiB")
        print(f"Allocated CUDA memory: {allocated_memory / (1024 ** 3):.2f} GiB")
        print(f"Free CUDA memory: {free_memory / (1024 ** 3):.2f} GiB")
    else:
        print("CUDA is not available.")
        
