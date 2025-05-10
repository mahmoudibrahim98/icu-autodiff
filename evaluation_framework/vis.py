import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import evaluation_framework.model_zoo as model_zoo
import evaluation_framework.SyntheticTimeSeriesEvaluator as SyntheticTimeSeriesEvaluator
import evaluation_framework.metric_utils as metric_utils
import torch
# from tqdm.notebook import trange
from tqdm import trange

from IPython.display import display
import os
import evaluation_framework.gru_evaluator as gru_evaluator
from datetime import datetime

def plot_distribution_comparisons(real_data, generated_data,features_names,subset):
    """
    real_data: shape (N, 25, 10)
    generated_data: shape (N, 25, 10)
    """
    plt.close('all')

    n_features = real_data.shape[2]
    
    # Determine the number of rows and columns based on the number of features
    n_cols = 6
    n_rows = (n_features + n_cols - 1) // n_cols  # This ensures enough rows to fit all features

    # Plot distributions for each feature
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes = axes.ravel()
    
    
    for i in range(n_features):
        # Plot real data distribution
        sns.kdeplot(real_data[:, :, i].flatten(), ax=axes[i], label='Real')
        # Plot generated data distribution
        sns.kdeplot(generated_data[:, :, i].flatten(), ax=axes[i], label='Generated')
        axes[i].set_title(features_names[i])
        axes[i].legend()
        
    plt.suptitle(f'Distribution Comparison of Real and Generated Data ({subset})')
    plt.tight_layout()
    return fig

def plot_temporal_statistics(real_data, generated_data, features_names, subset):
    """Plot mean and std over time for each feature"""
    n_features = real_data.shape[2]
    time_steps = np.arange(real_data.shape[1])
    
    # Determine the number of rows and columns based on the number of features
    n_cols = 6
    n_rows = (n_features + n_cols - 1) // n_cols  # This ensures enough rows to fit all features

    # Plot distributions for each feature
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes = axes.ravel()
    
    for i in range(n_features):
        # Calculate statistics
        real_mean = np.mean(real_data[:, :, i], axis=0)
        real_std = np.std(real_data[:, :, i], axis=0)
        gen_mean = np.mean(generated_data[:, :, i], axis=0)
        gen_std = np.std(generated_data[:, :, i], axis=0)
        
        # Plot means
        axes[i].plot(time_steps, real_mean, label='Real Mean')
        axes[i].plot(time_steps, gen_mean, label='Generated Mean')
        
        # Plot std bands
        axes[i].fill_between(time_steps, 
                            real_mean - real_std, 
                            real_mean + real_std, 
                            alpha=0.2)
        axes[i].fill_between(time_steps, 
                            gen_mean - gen_std, 
                            gen_mean + gen_std, 
                            alpha=0.2)
        
        axes[i].set_title(features_names[i])
        axes[i].legend()
    
    plt.suptitle(f'Temporal Statistics of Real and Generated Data ({subset})')
    plt.tight_layout()
    return fig

from scipy import stats

def compare_distributions(real_data, generated_data, features_names):
    """Compare distributions using KS test"""
    n_features = real_data.shape[2]
    results = []
    
    for i in range(n_features):
        # Perform Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(real_data[:, :, i].flatten(),
                                          generated_data[:, :, i].flatten())
        
        # Determine if the distributions are statistically different
        statistically_different = p_value < 0.05
        
        results.append({
            'feature': features_names[i],
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'statistically_different': statistically_different,
            'real_mean': np.mean(real_data[:, :, i]),
            'real_std': np.std(real_data[:, :, i]),
            'gen_mean': np.mean(generated_data[:, :, i]),
            'gen_std': np.std(generated_data[:, :, i])
        })
    
    return pd.DataFrame(results)

def plot_correlation_matrices(real_data, generated_data,features_names,subset):
    """Compare feature correlations between real and generated data"""
    # Calculate mean correlations across time steps
    real_corr = np.mean([np.corrcoef(real_data[:, t, :].T) 
                        for t in range(real_data.shape[1])], axis=0)
    gen_corr = np.mean([np.corrcoef(generated_data[:, t, :].T) 
                       for t in range(generated_data.shape[1])], axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.heatmap(real_corr, ax=ax1, cmap='coolwarm', vmin=-1, vmax=1)
    ax1.set_title('Real Data Correlations')
    
    sns.heatmap(gen_corr, ax=ax2, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title('Generated Data Correlations')
    plt.suptitle(f'Correlation Matrices of Real and Generated Data ({subset})')

    plt.tight_layout()
    return fig
def find_tstr_timestamp(out_path,iter_index):
    for entry in os.listdir(out_path):
        if entry.endswith(f"_iter{iter_index}"):  # e.g. 
            timestamp = entry.split("_")[0]
            return timestamp
def evaluate_synthetic_data(real_data, synth_data_list, feature_names, subset, real_data_y=None,
                            synth_data_y_list=None, indep_real=None, indep_real_y=None, demographic_data=None,
                            demographic_data_indep=None,
                            evaluations=['dist', 'temp', 'stat', 'pca', 'disc', 'temp_disc', 'pred', 'corr', 'trts', 'tstr', 'intersectional'],
                            intersectional = False, **kwargs):
    """
    Evaluate synthetic data quality using multiple metrics.
    
    Args:
        real_data: Real data tensor
        synth_data_list: List of synthetic data tensors to evaluate
        feature_names: Names of features
        subset: String indicating the subset being evaluated
        real_data_y: Real labels if available
        synth_data_y_list: List of synthetic labels corresponding to synth_data_list
        indep_real: Independent real data for evaluation
        indep_real_y: Independent real labels
        demographic_data: Static demographic data
        evaluations: List of evaluation metrics to compute
        **kwargs: Additional arguments for specific evaluations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_scores = []
    # tensor_cond = demographic_data.clone().detach().to(device)

    # Convert real data to numpy for plotting
    # temp_real = real_data.cpu().numpy()
    temp_real = real_data
    # Convert synthetic data list to numpy arrays
    # temp_synth_list = [s.cpu().numpy() for s in synth_data_list]
    temp_synth_list = synth_data_list
    # if 'dist' in evaluations:
    #     # Plot distribution for each synthetic dataset
    #     for i, temp_synth in enumerate(temp_synth_list):
    #         vis.plot_distribution_comparisons(temp_real, temp_synth, feature_names, f"{subset}_iter{i}")
    
    # if 'temp' in evaluations:
    #     # Plot temporal statistics for each synthetic dataset
    #     for i, temp_synth in enumerate(temp_synth_list):
    #         vis.plot_temporal_statistics(temp_real, temp_synth, feature_names, f"{subset}_iter{i}")
    
    # if 'stat' in evaluations:
    #     # Compute statistics for each synthetic dataset
    #     all_stats = []
    #     for temp_synth in temp_synth_list:
    #         df = vis.compare_distributions(temp_real, temp_synth, feature_names)
    #         all_stats.append(df)
    #     # Aggregate statistics
    #     results['stat'] = all_stats
    
    if 'pca' in evaluations:
        pca_size = kwargs.get('pca_sample_size', real_data.shape[0])
        pca_size = min([pca_size, real_data.shape[0]])
        pca_use_plotly = kwargs.get('pca_use_plotly', False)
        # Plot PCA for each synthetic dataset
        for i, temp_synth in enumerate(temp_synth_list):
            metric_utils.visualization(temp_real, temp_synth, 'pca', use_plotly=pca_use_plotly, compare=pca_size)

    if 'disc' in evaluations:
        '''
            {
        'synth_0': {
            'iter_0': {'overall': 0.1234, 'subgroup_1': 0.1100, 'subgroup_2': 0.1150},
            'iter_1': {'overall': 0.1156, 'subgroup_1': 0.1080, 'subgroup_2': 0.1120},
            ...
        },
        'synth_1': {
            'iter_0': {'overall': 0.0987, 'subgroup_1': 0.0950, 'subgroup_2': 0.0965},
            'iter_1': {'overall': 0.0876, 'subgroup_1': 0.0840, 'subgroup_2': 0.0855},
            ...
        }
        
        '''
        
        disc_iterations = kwargs.get('disc_iterations', 1000)
        disc_test_size = kwargs.get('disc_test_size', 0.3)
        disc_random_seed = kwargs.get('disc_random_seed', 42)
        disc_verbose = kwargs.get('disc_verbose', False)
        disc_counts = kwargs.get('disc_counts', 20)
        no, seq_len, dim = synth_data_list[0].shape
        hidden_dim = int(dim / 2)
        num_layers = 2
        for j,synth_data in enumerate(synth_data_list):
            # tensor_x_real = real_data.clone().detach().to(device)
            # tensor_x_synth = synth_data.clone().detach().to(device)
            tensor_x_real = real_data
            tensor_x_synth = synth_data
            for i in trange(disc_counts):
                if i == 0 and j == 0:
                    print("#### Discriminative scores: ####")
                discriminator = model_zoo.GRUDiscriminator(dim, hidden_dim, num_layers)
                metric = SyntheticTimeSeriesEvaluator.DiscriminativeMetric(discriminator)
                discriminative_score = metric(tensor_x_real, tensor_x_synth,demographic_data,intersectional=intersectional, test_size=disc_test_size, iterations=disc_iterations,
                                           random_seed=disc_random_seed, metadata=None, verbose=disc_verbose)
                
                all_scores.append({
                    'metric': 'discriminative',
                    'evaluation': 'discriminative',
                    'synth_index': j,
                    'iter_index': i,
                    **(discriminative_score if isinstance(discriminative_score, dict) else {'overall': discriminative_score})
                })

                if not intersectional:
                    print(f'Synth Data {j}, Iter {i}: {discriminative_score:.4f},', end=' ')
                else:
                    print(f'Synth Data {j}, Iter {i}, overall : {discriminative_score["overall"]:.4f},', end=' ')
        print()


    if 'pred' in evaluations:
        pred_iterations = kwargs.get('pred_iterations', 5000)
        pred_verbose = kwargs.get('pred_verbose', False)
        pred_counts = kwargs.get('pred_counts', 20)
        no, seq_len, dim = synth_data_list[0].shape
        hidden_dim = int(dim / 2)
        num_layers = 2
        
        for j,synth_data in enumerate(synth_data_list):
            for i in trange(pred_counts):
                if i == 0 and j == 0:
                    print("#### Predictive scores: ####")
                predictor = model_zoo.GRUPredictor(dim, hidden_dim)
                metric = SyntheticTimeSeriesEvaluator.PredictiveScoreMetric(predictor)
                predictive_score = metric(real_data, synth_data,demographic_data, pred_iterations, verbose=pred_verbose,intersectional=intersectional)
                all_scores.append({
                    'metric': 'predictive',
                    'evaluation': 'predictive',
                    'synth_index': j,
                    'iter_index': i,
                    **(predictive_score if isinstance(predictive_score, dict) else {'overall': predictive_score})
                })

                if not intersectional:
                    print(f'Synth Data {j}, Iter {i}: {predictive_score:.4f},', end=' ')
                else:
                    print(f'Synth Data {j}, Iter {i}, overall : {predictive_score["overall"]:.4f},', end=' ')
        print()
        # mean, sigma = metric_utils.display_scores(all_predictive_scores)
        # results['pred'] = {'mean': mean, 'sigma': sigma}

    # if 'temp_disc' in evaluations:
    #     temp_disc_iterations = kwargs.get('temp_disc_iterations', 1000)
    #     temp_disc_test_size = kwargs.get('temp_disc_test_size', 0.3)
    #     temp_disc_random_seed = kwargs.get('temp_disc_random_seed', 42)
    #     temp_disc_verbose = kwargs.get('temp_disc_verbose', False)
    #     temp_disc_counts = kwargs.get('temp_disc_counts', 20)
    #     no, seq_len, dim = synth_data_list[0].shape
    #     hidden_dim = int(dim / 2)
    #     num_layers = 2
        
    #     all_temp_discriminative_scores = []
    #     for j,synth_data in enumerate(synth_data_list):
    #         tensor_x_real = real_data.clone().detach().to(device)
    #         tensor_x_synth = synth_data.clone().detach().to(device)
    #         for i in trange(temp_disc_counts):
    #             temp_discriminator = model_zoo.GRUDiscriminator(dim, hidden_dim, num_layers)
    #             temp_metric = SyntheticTimeSeriesEvaluator.TemporalDiscriminativeMetric(temp_discriminator)
    #             temp_discriminative_score = temp_metric(tensor_x_real, tensor_x_synth, test_size=temp_disc_test_size, iterations=temp_disc_iterations,
    #                                                   random_seed=temp_disc_random_seed, metadata=None, verbose=temp_disc_verbose)
    #             all_temp_discriminative_scores.append(temp_discriminative_score)
    #             if len(all_temp_discriminative_scores) == 1:
    #                 print("#### Temporal Discriminative scores: ####")
    #             print(f'Synth Data {j}, Iter {i}: {temp_discriminative_score:.4f},', end=' ')
    #     print()
    #     mean, sigma = metric_utils.display_scores(all_temp_discriminative_scores)
    #     results['temp_disc'] = {'mean': mean, 'sigma': sigma}

    # if 'corr' in evaluations:
    #     corr_iterations = kwargs.get('corr_iterations', 5)
    #     corr_verbose = kwargs.get('corr_verbose', False)
        
    #     all_correlational_scores = []
    #     for j,synth_data in enumerate(synth_data_list):
    #         tensor_x_real = real_data.clone().detach().to(device)
    #         tensor_x_synth = synth_data.clone().detach().to(device)
    #         size = int(tensor_x_real.shape[0] / corr_iterations)
            
    #         def random_choice(size, num_select=100):
    #             select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    #             return select_idx
            
    #         for i in range(corr_iterations):
    #             real_idx = random_choice(tensor_x_real.shape[0], size)
    #             fake_idx = random_choice(tensor_x_synth.shape[0], size)
    #             corr = SyntheticTimeSeriesEvaluator.CrossCorrelLossMetric(tensor_x_real[real_idx, :, :])
    #             loss = corr(tensor_x_synth[fake_idx, :, :])
    #             all_correlational_scores.append(loss)
    #             if len(all_correlational_scores) == 1:
    #                 print("#### Correlational scores: ####")
    #             print(f'Synth Data {j}, Iter {i}: {loss:.4f},', end=' ')
    #     print()
    #     mean, sigma = metric_utils.display_scores(all_correlational_scores)
    #     results['corr'] = {'mean': mean, 'sigma': sigma}

    if 'trts' in evaluations:
        trts_models = kwargs.get('trts_models', None)
        trts_metrics = kwargs.get('trts_metrics', ['auroc'])
        trts_abs = True
        

        for j, (synth_data, synth_data_y) in enumerate(zip(synth_data_list, synth_data_y_list)):
            for i, model in enumerate(trts_models):
                if i == 0 and j == 0:
                    print("#### TRTS scores: ####")
                metric = SyntheticTimeSeriesEvaluator.TRTS(model)
                results = metric(real_data, real_data_y, synth_data, synth_data_y, demographic_data,
                                intersectional=intersectional,
                                trts_metrics=trts_metrics, trts_abs=trts_abs)
                # if isinstance(results, dict): 
                #     print('trts_score is a dict')
                for metric_name,types in results.items():
                    for type,value in types.items(): # type is rpd, real, synth
                        all_scores.append({
                            'metric': metric_name,
                            'evaluation': type,
                            'synth_index': j,
                            'iter_index': i,
                            **(value if isinstance(value, dict) else {'overall': value})
                        })
                trts_score = results['auroc']['trts_rpd']
                real_score = results['auroc']['trts_real']
                synth_score = results['auroc']['trts_synth']

                if not intersectional:

                    print(f'Synth Data {j}, Iter {i}: {trts_score:.4f}/{real_score:.4f}/{synth_score:.4f},', end=' ')

                else:
                    print(f"Synth Data {j}, Iter {i}: {trts_score['overall']:.4f}/{real_score['overall']:.4f}/{synth_score['overall']:.4f},", end=' ')
                
        print()

        
    if 'tstr' in evaluations:
        tstr_diff_timestamps = kwargs.get('tstr_diff_timestamps', '')
        print(tstr_diff_timestamps)
        tstr_model_params = kwargs.get('tstr_model_params', None)
        base_out_path = tstr_model_params['out_path']
        tstr_model_counts = kwargs.get('tstr_model_counts', 20)
        tstr_metrics = kwargs.get('tstr_metrics', ['auroc'])
        tstr_abs = kwargs.get('tstr_abs', True)
        tstr_preprocess_standardize = kwargs.get('tstr_preprocess_standardize', True)
        load = kwargs.get('load', False)
        results_real_dict = {}
        for i in range(tstr_model_counts):
            results_real_dict[i] = None
        for j, (synth_data, synth_data_y) in enumerate(zip(synth_data_list, synth_data_y_list)):
            for i in range(tstr_model_counts):
                if load:
                    timestamp = find_tstr_timestamp(os.path.join(base_out_path, tstr_diff_timestamps, f'gru/tstr/synth_{j}/'),i)
                else:
                    timestamp = datetime.now().strftime("%Y%m%d%H")
                tstr_model_params['out_path'] = os.path.join(base_out_path, tstr_diff_timestamps, f'gru/tstr/synth_{j}/{timestamp}_iter{i}')
                model = gru_evaluator.GRUEvaluator(tstr_model_params)
                metric = SyntheticTimeSeriesEvaluator.TSTR(model)
                train_real = j==0
                results = metric(real_data, real_data_y, synth_data, synth_data_y, indep_real, indep_real_y,
                        demographic_data, demographic_data_indep, intersectional=intersectional,
                        tstr_metrics=tstr_metrics, train_real=train_real,
                        preprocess_standardize=tstr_preprocess_standardize, load = load, results_real = results_real_dict[i])
                if train_real:
                    results_real_dict[i] = results
                tstr_model_params['out_path'] = base_out_path  # Reset to original out_path
                for metric_name,types in results.items():
                    for type,value in types.items(): # type is rpd, real, synth
                        all_scores.append({
                            'metric': metric_name,
                            'evaluation': type,
                            'synth_index': j,
                            'iter_index': i,
                            **(value if isinstance(value, dict) else {'overall': value})
                        })
                real_score = results['auroc']['tstr_real']
                synth_score = results['auroc']['tstr_synth']
                rpd_score = results['auroc']['tstr_rpd']
                if i == 0 and j == 0:
                    print("#### TSTR scores: ####")
                if not intersectional:
                    print(f'Synth Data {j}, Iter {i}: {rpd_score:.4f}/{real_score:.4f}/{synth_score:.4f},')

                else:
                    print(f"Synth Data {j}, Iter {i}: {rpd_score['overall']:.4f}/{real_score['overall']:.4f}/{synth_score['overall']:.4f}")


            print(f'\n')

        print()

    if 'mmd' in evaluations:

        compute_bandwidth = kwargs.get('mmd_compute_bandwidth', False)
        mmd_counts = kwargs.get('mmd_counts', 20)
        no, seq_len, dim = synth_data_list[0].shape
        hidden_dim = int(dim / 2)
        num_layers = 2

        for j,synth_data in enumerate(synth_data_list):
            for i in trange(mmd_counts):
                if i == 0 and j == 0:
                    print("#### MMD scores: ####")
                metric = SyntheticTimeSeriesEvaluator.MMDMetric()
                real_data_tensor = torch.tensor(real_data)
                synthetic_data_tensor = torch.tensor(synth_data)                
                mmd_score = metric(real_data_tensor, synthetic_data_tensor,demographic_data,intersectional=intersectional, compute_bandwidth = compute_bandwidth)
                all_scores.append({
                    'metric': 'MMD',
                    'evaluation': 'mmd',
                    'synth_index': j,
                    'iter_index': i,
                    **(mmd_score if isinstance(mmd_score, dict) else {'overall': mmd_score})
                })

                if not intersectional:
                    print(f'Synth Data {j}, Iter {i}: {mmd_score:.4f},', end=' ')
                else:
                    print(f'Synth Data {j}, Iter {i}, overall : {mmd_score["overall"]:.4f},', end=' ')
        print()
    return all_scores
        
def plot_patient_time_series_snapshots(real_data, _synth_data, all_features_names, output_dir, save=False, k=1, patient_indices=None, features_to_plot=None):
    """
    Plots time series snapshots for specified patients.

    Parameters:
    - real_data: tensor, containing the real data.
    - _synth_data: tensor, containing the synthetic data.
    - all_features_names: list, names of all the  features.
    - output_dir: str, directory to save the plots.
    - save: bool, whether to save the plots.
    - k: int, number of random patients to plot if patient_indices is None.
    - patient_indices: list, specific patient indices to plot. If None, k random patients will be selected.
    - features_to_plot: list, specific features to plot. If None, all important features will be plotted.
    """
    filepath = os.path.join(output_dir, 'figures')
    os.makedirs(filepath, exist_ok=True)

    if patient_indices is None:
        patient_indices = np.random.choice(real_data.shape[0], k, replace=False)

    if features_to_plot is None:
        features_to_plot = all_features_names
    else:
        features_to_plot = [feature for feature in features_to_plot if feature in all_features_names]

    num_features = len(features_to_plot)

    for label in patient_indices:
        fig, axes = plt.subplots((num_features + 2) // 3, 3, figsize=(20, 10))
        axes = axes.flatten()
        column_name = features_to_plot

        for i, feature in enumerate(column_name):
            feature_index = all_features_names.tolist().index(feature)
            axes[i].plot(real_data[label, :, feature_index].cpu(), marker='o', linestyle='-', color='b', label=f'Real {feature}')
            axes[i].plot(_synth_data[label, :, feature_index].numpy(), marker='o', linestyle='-', color='r', label=f'Conditionally Generated {feature}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(f'{feature}')
            axes[i].grid(True)
            axes[i].legend(loc='upper left')

        fig.suptitle(f'Patient {label} Time Series')
        plt.show()
        if save:
            fig.savefig(os.path.join(filepath, f'patient_{label}'))  # Save as PNG
