
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import abc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.metrics import mean_absolute_error
from evaluation_framework.metric_utils import extract_time
import copy
import pandas as pd
import os
# We should use something like this:
import numpy as np
import torch
from scipy.stats import gaussian_kde

def calculate_kl_divergence_kde(p_samples, q_samples, num_points=1000):
    """Calculate KL divergence using kernel density estimation (KDE)."""
    
    # Convert to numpy arrays if needed
    if isinstance(p_samples, torch.Tensor):
        p_samples = p_samples.cpu().numpy()
    else:
        p_samples = np.array(p_samples)
    
    if isinstance(q_samples, torch.Tensor):
        q_samples = q_samples.cpu().numpy()
    else:
        q_samples = np.array(q_samples)
    
    # Estimate probability densities using KDE
    p_kde = gaussian_kde(p_samples)
    q_kde = gaussian_kde(q_samples)
    
    # Create a range of points for evaluation
    x_min = min(np.min(p_samples), np.min(q_samples))
    x_max = max(np.max(p_samples), np.max(q_samples))
    x_eval = np.linspace(x_min, x_max, num_points)
    
    # Evaluate probability densities
    p_density = p_kde(x_eval)
    q_density = q_kde(x_eval)
    
    # Add a small constant to avoid division by zero
    p_density += 1e-10
    q_density += 1e-10
    
    # Compute KL divergence using numerical integration
    kl_divergence = np.sum(p_density * np.log(p_density / q_density) * (x_eval[1] - x_eval[0]))
    
    return kl_divergence


def calculate_kl_divergence(p_samples, q_samples, n_bins=50):
    """Calculate KL divergence between two sets of samples using histograms"""
    # Convert tensors to numpy arrays
    if isinstance(p_samples, torch.Tensor):
        p_samples = p_samples.cpu().numpy()
    else:
        p_samples = np.array(p_samples)
        
    if isinstance(q_samples, torch.Tensor):
        q_samples = q_samples.cpu().numpy()
    else:
        q_samples = np.array(q_samples)
    
    # Get the range for both distributions
    min_val = min(np.min(p_samples), np.min(q_samples))
    max_val = max(np.max(p_samples), np.max(q_samples))
    
    # Create histograms with same bins
    p_hist, bin_edges = np.histogram(p_samples, bins=n_bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=True)
    
    # Add small constant to avoid division by zero
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    
    # Calculate KL divergence
    return np.sum(p_hist * np.log(p_hist / q_hist))
class Metric(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass


def train_test_divide(ori_data, generated_data, demographic_data, test_size=0.2, seed = 42):
    if demographic_data is None:
        train_x, test_x = train_test_split(ori_data, test_size=test_size, random_state=seed)
        train_x_hat, test_x_hat = train_test_split(generated_data, test_size=test_size, random_state=seed)
        return train_x, train_x_hat, test_x, test_x_hat

    else:
        train_x, test_x,train_demographic_data, test_demographic_data = train_test_split(ori_data, demographic_data, test_size=test_size, random_state=seed,stratify=demographic_data)
        train_x_hat, test_x_hat,train_demographic_data_hat, test_demographic_data_hat = train_test_split(generated_data, demographic_data, test_size=test_size, random_state=seed,stratify=demographic_data)
        return train_x, train_x_hat, test_x, test_x_hat, train_demographic_data, test_demographic_data, train_demographic_data_hat, test_demographic_data_hat



class BaseDownstreamEvaluator(abc.ABC):
    def train_model(self, *args, **kwargs):
        pass
    def evaluate(self, *args, **kwargs):
        pass

class DiscriminativeMetric(Metric):
    """
    This metric measures the discriminative performance of a classification 
    model (optimizing a 2-layer LSTM) in distinguishing between sequences
    from the original and generated datasets..

    This metric evaluates a discriminative model by training it on a combination of synthetic
    and real datasets and assessing its performance on a test set (containing mixed synthetic and real).
    
    The lower the score, the better the performance.

    Args:

    Returns:
        float: Computed score if evaluate_subgroups is False, otherwise returns a dictionary with scores per subgroup.

    Example:
        >>> no, seq_len, dim = real_Data.shape
        >>> hidden_dim = int(dim / 2)
        >>> num_layers = 2
        >>> discriminator = GRUDiscriminator(dim, hidden_dim, num_layers)
        >>> metric = DiscriminativeMetric(discriminator)
        >>> result = metric(real_Data, synthetic_Data , test_size = 0.7 , random_seed = None,metadata = None)
        >>> print(result)
    """
    
    def __init__(self, discriminator: BaseDownstreamEvaluator) -> None:

        self._discriminator = discriminator

    
    def __call__(self, ori_data, gen_data,demographic_data,intersectional = False, test_size = 0.7 ,iterations = 1000,
                 random_seed = 42,metadata = None, verbose = False):
        # Prepare data
        if not intersectional:
            X_train, X_train_hat, X_test, X_test_hat = train_test_divide(ori_data, gen_data,demographic_data,test_size,random_seed)
        else:
            X_train, X_train_hat, X_test, X_test_hat, train_demographic_data, test_demographic_data, train_demographic_data_hat, test_demographic_data_hat = train_test_divide(ori_data, gen_data,demographic_data,test_size,random_seed)
        # print(f"Type of X_test: {type(X_test)},Device : {X_test.device}")
        # print(f"Type of X_test_hat: {type(X_test_hat)},Device : {X_test_hat.device}")

        # Train discriminator
        self._discriminator.train_model(X_train, X_train_hat, iterations=iterations, verbose=verbose) 
        
        # evaluate the discriminator  
        if not intersectional:
            discriminative_score = self._discriminator.evaluate(X_test, X_test_hat)
        else:
            """Evaluate discriminative performance for a specific subgroup using the trained discriminator,
            as well as the overall score"""
            discriminative_score = {}
            overall_discriminative_score = self._discriminator.evaluate(X_test, X_test_hat)
            discriminative_score['overall'] = overall_discriminative_score
            
            combinations = pd.DataFrame(demographic_data).drop_duplicates()
            # Evaluate each combination

            for _, row in combinations.iterrows():
                # Create mask for this demographic combination
                mask = np.ones(len(test_demographic_data), dtype=bool)
                mask_hat = np.ones(len(test_demographic_data_hat), dtype=bool)
                for col in combinations.columns:
                    mask &= test_demographic_data[:, col] == row[col]
                    mask_hat &= test_demographic_data_hat[:, col] == row[col]
                X_test_subgroup = X_test[mask]
                X_test_hat_subgroup = X_test_hat[mask_hat]
                subgroup_discriminative_score = self._discriminator.evaluate(X_test_subgroup, X_test_hat_subgroup)
                if subgroup_discriminative_score is not None:
                    subgroup_name = "_".join([f"{int(v)}" for k, v in row.items()])
                    discriminative_score[subgroup_name] = subgroup_discriminative_score
        
        return discriminative_score
def populate_metric_results(results, metrics, real_values, synth_values, trts_abs=True, prefix="trts_", calculate_rpd = True):
    """
    Populate results dictionary with metric values and their difference.
    
    Args:
        results: Dictionary to populate
        metric: Metric name
        real_value: Value from real data evaluation
        synth_value: Value from synthetic data evaluation
        trts_abs: Whether to use absolute difference (default: True)
        prefix: Prefix for keys in the results dictionary (default: "trts_")
        
    Returns:
        Updated results dictionary
    """
    for metric in metrics: 
        if metric not in results:
            results[metric] = {}
        
        results[metric][f'{prefix}real'] = real_values[metric]
        results[metric][f'{prefix}synth'] = synth_values[metric]
        if calculate_rpd:
            diff = abs(real_values[metric] - synth_values[metric]) if trts_abs else (real_values[metric] - synth_values[metric] )
            results[metric][f'{prefix}rpd'] = diff
        
    return results  

def populate_metric_results_dict(results, metrics, real_values, synth_values,subgroup, trts_abs=True,prefix="trts_", calculate_rpd = True):
    """
    Populate results dictionary with metric values and their difference.
    
    Args:
        results: Dictionary to populate
        metric: Metric name
        real_value: Value from real data evaluation
        synth_value: Value from synthetic data evaluation
        trts_abs: Whether to use absolute difference (default: True)
        prefix: Prefix for keys in the results dictionary (default: "trts_")
        
    Returns:
        Updated results dictionary
    """
    for metric in metrics: 
        if metric not in results:
            results[metric] = {}
            results[metric][f'{prefix}real'] = {}
            results[metric][f'{prefix}synth'] = {}
            if calculate_rpd:
                results[metric][f'{prefix}rpd'] = {}
        results[metric][f'{prefix}real'][subgroup] = real_values[metric]
        results[metric][f'{prefix}synth'][subgroup] = synth_values[metric]
        
        if calculate_rpd:
            if trts_abs:
                if real_values[metric] is not None and synth_values[metric] is not None:
                    diff =  abs(real_values[metric] - synth_values[metric])
                else:
                    # Handle the case where one of the values is None
                    diff = None  # or set to a default value, e.g., 0
            else:
                if real_values[metric] is not None and synth_values[metric] is not None:
                    diff =  abs(real_values[metric] - synth_values[metric])
                else:
                    # Handle the case where one of the values is None
                    diff = None  # or set to a default value, e.g., 0
            results[metric][f'{prefix}rpd'][subgroup] = diff
             
                        
    return results  
def populate_kl_divergence(results, real_preds_test, synth_preds_test, prefix="trts_"):
    kl_divergence = calculate_kl_divergence(real_preds_test, synth_preds_test)
    if 'kl_divergence' not in results:
        results['kl_divergence'] = {}
    results['kl_divergence'][prefix] = kl_divergence
    return results
def populate_kl_divergence_dict(results, real_preds_test, synth_preds_test, subgroup, prefix="trts_"):
    kl_divergence = calculate_kl_divergence(real_preds_test, synth_preds_test)
    if 'kl_divergence' not in results:
        results['kl_divergence'] = {}
    if prefix not in results['kl_divergence']:
        results['kl_divergence'][prefix] = {}
    results['kl_divergence'][prefix][subgroup] = kl_divergence
    return results  

class TRTS(Metric):
    """
    TRTS (Train Real Test Synthetic) metric class.
    This class computes the relative performance difference between real and synthetic data
    using a specified downstream evaluator and metric.
    
    The lower the better.
    
    Args:
        downstream_evaluator (BaseDownstreamEvaluator): An instance of a downstream evaluator trained on real data.
    
    Returns:
        dict: A nested dictionary with the following structure:
        
        Non-intersectional case (intersectional=False):
        {
            'metric_name1': {
                'trts_real': value,      # Metric value on real data
                'trts_synth': value,     # Metric value on synthetic data
                'trts_rpd': value        # Relative performance difference (absolute if trts_abs=True)
            },
            'metric_name2': {
                'trts_real': value,
                'trts_synth': value,
                'trts_rpd': value
            },
            ...
            'kl_divergence': {
                'trts_': value           # KL divergence between real and synthetic predictions
            }
        }
        
        Intersectional case (intersectional=True):
        {
            'metric_name1': {
                'trts_real': {
                    'overall': value,     # Overall metric value on real data
                    'subgroup1_name': value,
                    'subgroup2_name': value,
                    ...
                },
                'trts_synth': {
                    'overall': value,     # Overall metric value on synthetic data
                    'subgroup1_name': value,
                    'subgroup2_name': value,
                    ...
                },
                'trts_rpd': {
                    'overall': value,     # Overall relative performance difference
                    'subgroup1_name': value,
                    'subgroup2_name': value,
                    ...
                }
            },
            ...
            'kl_divergence': {
                'trts_': {
                    'overall': value,     # Overall KL divergence
                    'subgroup1_name': value,
                    'subgroup2_name': value,
                    ...
                }
            }
        }
        
        Where subgroup names are created by joining demographic feature values with underscores
        (e.g., "0_1" for demographic values 0 and 1).

    Example:
        >>> trained_model = trained_model()
        >>> metric = TRTS(trained_model)
        >>> result = metric(real_Data,real_Data_y, synthetic_Data,synthetic_Data_y, metric = 'auroc')
        >>> print(result)
    """
    
    def __init__(self, downstream_evaluator: BaseDownstreamEvaluator) -> None:

        self.downstream_evaluator = downstream_evaluator


    def __call__(self, ori_data,ori_data_y, gen_data,gen_data_y,demographic_data, intersectional = False,
                 trts_metrics = ['auroc'], trts_abs = True):
        # results ={} trts_rpd, trts_real, trts_synth,
        results = {}
        if not intersectional:
            eval_real = self.downstream_evaluator.evaluate(ori_data, ori_data_y)
            eval_synthetic = self.downstream_evaluator.evaluate(gen_data, gen_data_y)
            results = populate_metric_results(results,
                                    trts_metrics,
                                    eval_real,
                                    eval_synthetic,
                                    trts_abs,
                                    prefix='trts_')
            real_preds_test, real_labels_test = self.downstream_evaluator.predict_test(ori_data, ori_data_y)
            synth_preds_test, synth_labels_test = self.downstream_evaluator.predict_test(gen_data, gen_data_y)
            results = populate_kl_divergence(results, real_preds_test, synth_preds_test, prefix='trts_')
        else:
            eval_real = self.downstream_evaluator.evaluate(ori_data, ori_data_y)
            eval_synthetic = self.downstream_evaluator.evaluate(gen_data, gen_data_y)
            results = populate_metric_results_dict(results,
                                    trts_metrics,
                                    eval_real,
                                    eval_synthetic,
                                    'overall',
                                    trts_abs,
                                    prefix='trts_')
            real_preds_test, real_labels_test = self.downstream_evaluator.predict_test(ori_data, ori_data_y)
            synth_preds_test, synth_labels_test = self.downstream_evaluator.predict_test(gen_data, gen_data_y)
            results = populate_kl_divergence_dict(results, real_preds_test, synth_preds_test, 'overall', prefix='trts_')
            
            combinations = pd.DataFrame(demographic_data).drop_duplicates()
            # Evaluate each combination
            for _, row in combinations.iterrows():
                # Create mask for this demographic combination
                mask = np.ones(len(demographic_data), dtype=bool)
                for col in combinations.columns:
                    mask &= demographic_data[:, col] == row[col]
                X_test_subgroup = ori_data[mask]
                y_test_subgroup = ori_data_y[mask]
                X_test_hat_subgroup = gen_data[mask]
                y_test_hat_subgroup = gen_data_y[mask]
                subgroup_name = "_".join([f"{int(v)}" for k, v in row.items()])
                eval_real = self.downstream_evaluator.evaluate(X_test_subgroup, y_test_subgroup)
                eval_synthetic = self.downstream_evaluator.evaluate(X_test_hat_subgroup, y_test_hat_subgroup)
                results = populate_metric_results_dict(results,
                                    trts_metrics,
                                    eval_real,
                                    eval_synthetic,
                                    subgroup_name,
                                    trts_abs,
                                    prefix='trts_')
                
                subgroup_real_preds_test, subgroup_real_labels_test = self.downstream_evaluator.predict_test(X_test_subgroup, y_test_subgroup)
                subgroup_synth_preds_test, subgroup_synth_labels_test = self.downstream_evaluator.predict_test(X_test_hat_subgroup, y_test_hat_subgroup)
                results = populate_kl_divergence_dict(results, subgroup_real_preds_test, subgroup_synth_preds_test,
                                                      subgroup_name, prefix='trts_')
                

        return results
def find_tstr_timestamp(out_path,type):
    for entry in os.listdir(out_path):
        if entry.startswith(type):  # e.g. 
            return entry
class TSTR(Metric):
    """
    TRTS (Train Synthetic Test Real) metric class.
    This class trains two donwstream evaluators, first on real and other on synthetic data
    The models are then evaluated on indep real data.
        
    The lower the better.
    
    Args:
        downstream_evaluator (BaseDownstreamEvaluator): An instance of a downstream evaluator to be trained on synthetic data.
    
    Returns:
        float: Computed .

    Example:
        >>> model = model()
        >>> metric = TSTR(model)
        >>> result = metric(real_Data,real_Data_y, synthetic_Data,synthetic_Data_y,metadata, metric = 'auroc')
        >>> print(result)
    """
    
    def __init__(self, downstream_evaluator: BaseDownstreamEvaluator) -> None:

        self.downstream_evaluator_real = downstream_evaluator
        self.downstream_evaluator_synth = copy.deepcopy(downstream_evaluator)
        self.dim = downstream_evaluator.input_size
        self.out_path = downstream_evaluator.out_path
    def __call__(self, ori_data,ori_data_y, gen_data,gen_data_y, indep_real, indep_real_y,
                 demographic_data,demographic_data_indep, intersectional = False, tstr_metrics = ['auroc'],
                 train_real = False, preprocess_standardize = True, load = False, results_real = None):
        results = {}
        train_idx, test_idx = train_test_split(np.arange(gen_data.shape[0]), test_size=0.2,)
        synth_train_X = gen_data[train_idx]
        synth_train_y = gen_data_y[train_idx]
        synth_val_X = gen_data[test_idx]
        synth_val_y = gen_data_y[test_idx]
        real_train_X = ori_data[train_idx]
        real_train_y = ori_data_y[train_idx]
        real_val_X = ori_data[test_idx]
        real_val_y = ori_data_y[test_idx]

        synth_train_X, synth_val_X, synth_train_y, synth_val_y = train_test_split(gen_data, gen_data_y, test_size=0.2)
        real_train_X, real_val_X, real_train_y, real_val_y = train_test_split(ori_data, ori_data_y, test_size=0.2)
        metadata_real =  f'real_{self.dim}features'
        metadata_synth =  f'synth_{self.dim}features'

        if load:
            real_path = find_tstr_timestamp(self.out_path, 'real')
            real_timestamp = real_path.split('_')[1]
            synth_path = find_tstr_timestamp(self.out_path, 'synth')
            synth_timestamp = synth_path.split('_')[1]
            real_out_path = os.path.join(self.out_path, real_path, f'{metadata_real}_{real_timestamp}.pt')
            synth_out_path = os.path.join(self.out_path, synth_path, f'{metadata_synth}_{synth_timestamp}.pt')
            self.downstream_evaluator_real.load(real_out_path)
            self.downstream_evaluator_synth.load(synth_out_path)
            self.downstream_evaluator_real.preprocess_standardize = True
            self.downstream_evaluator_synth.preprocess_standardize = True
        else:
            if train_real:
                print('Training model on real data')
                self.downstream_evaluator_real.train(real_train_X, real_val_X, real_train_y , real_val_y,
                                                     metadata_real,preprocess_standardize)
            print('Training model on synthetic data')
            self.downstream_evaluator_synth.train(synth_train_X, synth_val_X, synth_train_y , synth_val_y,
                                                  metadata_synth,preprocess_standardize)
            

        
        
        if not intersectional:
            if train_real:
                eval_real = self.downstream_evaluator_real.evaluate(indep_real, indep_real_y)
            else:
                eval_real = {}
                for metric,values in results_real.items():
                    eval_real[metric] = values['tstr_real']
            eval_synth = self.downstream_evaluator_synth.evaluate(indep_real, indep_real_y)
            results = populate_metric_results(results,
                                    tstr_metrics,
                                    eval_real,
                                    eval_synth,
                                    prefix='tstr_')
        else:

            combinations = pd.DataFrame(demographic_data_indep).drop_duplicates()
            # Evaluate each combination
            if train_real:
                eval_real = self.downstream_evaluator_real.evaluate(indep_real, indep_real_y)
            else:
                eval_real = {}
                for metric,values in results_real.items():
                    eval_real[metric] = values['tstr_real']['overall']
            eval_synth = self.downstream_evaluator_synth.evaluate(indep_real, indep_real_y)
            results = populate_metric_results_dict(results,
                                    tstr_metrics,
                                    eval_real,
                                    eval_synth,
                                    'overall',
                                    prefix='tstr_')

            for _, row in combinations.iterrows():
                # Create mask for this demographic combination
                mask = np.ones(len(demographic_data_indep), dtype=bool)
                for col in combinations.columns:
                    mask &= demographic_data_indep[:, col] == row[col]
                indep_real_subgroup = indep_real[mask]
                indep_real_y_subgroup = indep_real_y[mask]
                subgroup_name = "_".join([f"{int(v)}" for k, v in row.items()])
                if train_real:
                    eval_real = self.downstream_evaluator_real.evaluate(indep_real_subgroup, indep_real_y_subgroup)
                else:
                    eval_real = {}
                    for metric,values in results_real.items():
                        eval_real[metric] = values['tstr_real'][subgroup_name]
                eval_synth = self.downstream_evaluator_synth.evaluate(indep_real_subgroup, indep_real_y_subgroup)
                results = populate_metric_results_dict(results,
                                    tstr_metrics,
                                    eval_real,
                                    eval_synth,
                                    subgroup_name,
                                    prefix='tstr_')

        return results
    
class TemporalDiscriminativeMetric(Metric):
    """
    This metric measures the similarity of distributions of inter-row differences 
    between generated and original sequential data  to check if the generated 
    data preserves the temporal dependencies of the original data.
        
    This metric evaluates a discriminative model by training it over the differenced 
    matrices from original and synthetic data.

    We average discriminative scores over k randomly selected t ∈ {1, . . . , T − 1}.
  
    The lower the score, the better the performance.
    
    Args:

    Returns:
        float: Computed Temporal Discriminative Score .

    Example:
        >>> no, seq_len, dim = real_Data.shape
        >>> hidden_dim = int(dim / 2)
        >>> num_layers = 2
        >>> discriminator = LSTMDiscriminator(dim, hidden_dim, num_layers)
        >>> metric = TemporalDiscriminativeMetric(discriminator)
        >>> result = metric(real_Data, synthetic_Data , test_size = 0.7 , random_seed = None,metadata = None)
        >>> print(result)
    """
    
    def __init__(self, discriminator: BaseDownstreamEvaluator) -> None:

        self._discriminator = discriminator


    def __call__(self, ori_data, gen_data,test_size = 0.7 ,iterations = 1000, random_seed = None,metadata = None, verbose = False):
       # 2 sources of randomness: 1. randint 2. Random seed for train-test split of the discrimnator.
        result = []; B, T, L = ori_data.shape;    
        t = torch.randint(1, T, (1,)).item()

        diff_real = torch.empty(B, T-t, L)
        diff_synth = torch.empty(B, T-t, L)

        for i in range(T-t):
            diff_real[:,i,:] = ori_data[:,i,:] - ori_data[:,i+t,:]
            diff_synth[:,i,:] = gen_data[:,i,:] - gen_data[:,i+t,:]
            
        discriminative_metric = DiscriminativeMetric(self._discriminator)
        temporal_score = discriminative_metric(diff_real, diff_synth, test_size, iterations, random_seed = random_seed, metadata = metadata, verbose = verbose)

        return temporal_score
    


class ConditionalPredictiveScore(Metric):
    """
    Return the area under the curve (AUC) scores of a
    classifier trained only using synthetic data. The classifier
    is trained to predict the metadata given the corresponding
    synthetic time series. We then test this classifier on the
    real unseen test dataset.

    A low prediction error indicates that the generated data indeed
    include features associated to the conditioning. 
    Args:

    Returns:
        float: Computed .

    Example:
        >>> no, seq_len, dim = real_Data.shape
        >>> output_size = ori_labels.shape[1]
        >>> hidden_dim = int(dim / 2)
        >>> num_layers = 2
        >>> evaluator = LSTMCondtionalEvaluator(dim, hidden_dim, num_layers, output_size)
        >>> metric = ConditionalPredictiveScore(evaluator)
        >>> result = metric(real_Data, synthetic_Data , test_size = 0.7 , random_seed = None,metadata = None)
        >>> print(result)
    """
    
    def __init__(self, evaluator: BaseDownstreamEvaluator) -> None:

        self._evaluator = evaluator


    def __call__(self, ori_data, ori_labels, synthetic_data, synthetic_labels,iterations = 1000, random_seed = None,metadata = None):

        self._evaluator.train_model(synthetic_data, synthetic_labels, iterations=iterations)

        roc_auc = self._evaluator.evaluate(ori_data, ori_labels)

        return roc_auc
    
 
class CrossCorrelLossMetric(Metric):
    """
    This metric calculates the cross-correlation loss between real and synthetic data.

    The lower, the better.
    Args:

    Returns:
        float: Computed Cross-Correlation Loss.

    Example:
        >>> no, seq_len, dim = real_Data.shape
        >>> metric = CrossCorrelLossMetric(real_Data)
        >>> result = metric(real_Data, synthetic_Data)
        >>> print(result)
    """
    
    def __init__(self, real_data, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        self.real_data = real_data
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo
        self.cross_correl_real = self.cacf_torch(self.transform(real_data), 1).mean(0)[0]

    def cacf_torch(self, x, max_lag, dim=(0, 1)):
        def get_lower_triangular_indices(n):
            return [list(x) for x in torch.tril_indices(n, n)]

        ind = get_lower_triangular_indices(x.shape[2])
        x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
        x_l = x[..., ind[0]]
        x_r = x[..., ind[1]]
        cacf_list = list()
        for i in range(max_lag):
            y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
            cacf_i = torch.mean(y, (1))
            cacf_list.append(cacf_i)
        cacf = torch.cat(cacf_list, 1)
        return cacf.reshape(cacf.shape[0], -1, len(ind[0]))

    def compute(self, x_fake):
        cross_correl_fake = self.cacf_torch(self.transform(x_fake), 1).mean(0)[0]
        loss = self.norm_foo(cross_correl_fake - self.cross_correl_real.to(x_fake.device))
        return loss / 10.

    def __call__(self, synthetic_data) -> float:
        loss_componentwise = self.compute(synthetic_data)
        return self.reg * loss_componentwise.mean().item()

def dimension_wise_probability(self):
    n_dimensions = self.real_data.shape[1]
    ks_stats = []

    for dim in range(n_dimensions):
        real_dim = self.real_data[:, dim]
        synthetic_dim = self.synthetic_data[:, dim]
        ks_stat, _ = ks_2samp(real_dim, synthetic_dim)
        ks_stats.append(ks_stat)

    return ks_stats

def evaluate(self):
    mmd = self.maximum_mean_discrepancy()
    dwp = self.dimension_wise_probability()
    ds = self.discriminative_score()

    return {
        "Maximum Mean Discrepancy": mmd,
        "Dimension-wise Probability": dwp,
        "Discriminative Score": ds
    }

import torch
import numpy as np



class MMDMetric(Metric):
    """
    This metric calculates MMD between real and synthetic sequential samples.
    
    Handles sequential data with shape (N_samples, seq_len, feature_dim).
    """
    
    def __init__(self):
        self.bandwidths = torch.tensor([1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    
    def compute_bandwidths(self, data):
        """Compute kernel bandwidths using median heuristic."""
        with torch.no_grad():
            flattened_data = data.reshape(data.size(0), -1)
            pairwise_dists = torch.sum((flattened_data[:, None, :] - flattened_data[None, :, :]) ** 2, dim=-1)
            median_dist = torch.median(pairwise_dists)
            return torch.tensor([median_dist / 2, median_dist, median_dist * 2])

    def _mix_rbf_kernel(self, X, Y, sigmas, wts=None):
        """Compute mixed RBF kernel for sequential data."""
        if wts is None:
            wts = torch.ones(sigmas.shape[0], device=X.device)
        print('flattening data...')
        X_flat = X.reshape(X.size(0), -1)
        Y_flat = Y.reshape(Y.size(0), -1)
        print('computing einsum...')
        XX = torch.einsum('nd,md->nm', X_flat, X_flat)
        XY = torch.einsum('nd,md->nm', X_flat, Y_flat)
        YY = torch.einsum('nd,md->nm', Y_flat, Y_flat)
        print('computing sqnorms...')
        X_sqnorms = torch.diagonal(XX)
        Y_sqnorms = torch.diagonal(YY)
        
        K_XX, K_XY, K_YY = 0, 0, 0
        print('computing kernels...')
        for sigma, wt in zip(sigmas, wts):
            print(f'computing kernel for sigma={sigma} and wt={wt}...')
            gamma = 1 / (2 * sigma ** 2)
            K_XX += wt * torch.exp(-gamma * (-2 * XX + X_sqnorms.unsqueeze(0) + X_sqnorms.unsqueeze(1)))
            K_XY += wt * torch.exp(-gamma * (-2 * XY + X_sqnorms.unsqueeze(0) + Y_sqnorms.unsqueeze(1)))
            K_YY += wt * torch.exp(-gamma * (-2 * YY + Y_sqnorms.unsqueeze(0) + Y_sqnorms.unsqueeze(1)))
            
        return K_XX, K_XY, K_YY, wts.sum()

    def _mmd2(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        """Compute squared MMD value."""
        m = float(K_XX.size(0))
        n = float(K_YY.size(0))
        
        if biased:
            mmd2 = (K_XX.sum() / (m * m) + 
                    K_YY.sum() / (n * n) - 
                    2 * K_XY.sum() / (m * n))
        else:
            if const_diagonal is not False:
                trace_X = m * const_diagonal
                trace_Y = n * const_diagonal
            else:
                trace_X = torch.trace(K_XX)
                trace_Y = torch.trace(K_YY)
            
            mmd2 = ((K_XX.sum() - trace_X) / (m * (m - 1)) +
                    (K_YY.sum() - trace_Y) / (n * (n - 1)) -
                    2 * K_XY.sum() / (m * n))
            
        return mmd2

    def mix_rbf_mmd2(self, X, Y, sigmas, wts=None, biased=True):
        """Compute MMD using mixture of RBF kernels."""
        print("Computing MMD using mixture of RBF kernels...")
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigmas, wts)
        return self._mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    @torch.no_grad()
    def __call__(self, real_data: torch.Tensor, synthetic_data: torch.Tensor, demographic_data: torch.Tensor,intersectional: bool = False, compute_bandwidth: bool = False) -> float:
        """
        Compute MMD between real and synthetic sequential data.
        
        Args:
            real_data (torch.Tensor): Real data of shape (n_samples, seq_len, feature_dim)
            synthetic_data (torch.Tensor): Synthetic data of shape (n_samples, seq_len, feature_dim)
            compute_bandwidth (bool): Whether to compute bandwidth using median heuristic
            
        Returns:
            float: MMD value
        """
        # Verify input shapes
        assert real_data.dim() == 3, f"Expected 3D tensor (n_samples, seq_len, feature_dim), got shape {real_data.shape}"
        assert synthetic_data.dim() == 3, f"Expected 3D tensor (n_samples, seq_len, feature_dim), got shape {synthetic_data.shape}"
        assert real_data.shape[1:] == synthetic_data.shape[1:], "Sequence length and feature dimensions must match"
        
        if compute_bandwidth:
            print("Computing bandwidths...")

            self.bandwidths = self.compute_bandwidths(real_data)
            
        print("Calculating MMD...")
        if not intersectional:
            mmd2_value = self.mix_rbf_mmd2(
                real_data,
                synthetic_data,
                sigmas=self.bandwidths.to(real_data.device)
            )
            mmd_metric = float(torch.sqrt(torch.clamp(mmd2_value, min=0.0)).item())
        else:
            mmd_metric = {}
            # print("Calculating MMD for overall...")
            # overall_mmd2_value = self.mix_rbf_mmd2(
            #     real_data,
            #     synthetic_data,
            #     sigmas=self.bandwidths.to(real_data.device)
            # )
            # mmd_metric['overall'] = float(torch.sqrt(torch.clamp(overall_mmd2_value, min=0.0)).item())
            combinations = pd.DataFrame(demographic_data).drop_duplicates()
            print("Calculating MMD for subgroups...")
            for _, row in combinations.iterrows():
                mask = np.ones(len(demographic_data), dtype=bool)
                for col in combinations.columns:
                    mask &= demographic_data[:, col] == row[col]
                subgroup_name = "_".join([f"{int(v)}" for k, v in row.items()])
                real_data_subgroup = real_data[mask]
                synthetic_data_subgroup = synthetic_data[mask]
                if real_data_subgroup.shape[0] < 2 or synthetic_data_subgroup.shape[0] < 2:
                    print(f"Skipping subgroup {subgroup_name}: not enough samples.")
                    continue

                mmd2_value = self.mix_rbf_mmd2(real_data_subgroup, synthetic_data_subgroup, sigmas=self.bandwidths.to(real_data.device))
                print(f"MMD for {subgroup_name}: {mmd2_value}")
                mmd_metric[subgroup_name] = float(torch.sqrt(torch.clamp(mmd2_value, min=0.0)).item())
        return mmd_metric
class PredictiveScoreMetric(Metric):
    """

    PredictiveScoreMetric is a class that measures the utility of generated sequences by training a posthoc
    sequence prediction model (optimizing a 2-layer LSTM) to predict next-step temporal vectors under
    a Train-on-Synthetic-Test-on-Real (TSTR) framework. 
    
    Training is done on synthetic data, and evaluation on real data.

    The lower the score, the better the performance.    
    
    Args:
        predictor (BaseDownstreamEvaluator): An instance of a downstream evaluator used for training and evaluation.
    Returns:
        float: Mean Absolute Error (MAE) of the predictions on the original data.

    Example:
        >>> no, seq_len, dim = real_Data.shape
        >>> hidden_dim = int(dim / 2)
        >>> predictor = GRUPredictor(input_dim=dim, hidden_dim=hidden_dim)
        >>> metric = PredictiveScoreMetric(predictor)
        >>> result = metric(real_Data, synthetic_Data, iterations, verbose)
        >>> print(result)
        
    """

    def __init__(self, predictor: BaseDownstreamEvaluator) -> None:

        self._predictor = predictor

    def __call__(self, ori_data, generated_data,demographic_data, iterations,verbose, intersectional = False):

        # Train discriminator
        self._predictor.train_model(ori_data, generated_data, iterations=iterations, verbose=verbose) 
        
        # evaluate the discriminator    
        if not intersectional:
            predictive_score = self._predictor.evaluate(ori_data)
        else:
            predictive_score = {}
            overall_predictive_score = self._predictor.evaluate(ori_data)
            predictive_score['overall'] = overall_predictive_score

            combinations = pd.DataFrame(demographic_data).drop_duplicates()
            # Evaluate each combination

            for _, row in combinations.iterrows():
                # Create mask for this demographic combination
                mask = np.ones(len(demographic_data), dtype=bool)
                for col in combinations.columns:
                    mask &= demographic_data[:, col] == row[col]
                subgroup_predictive_score = self._predictor.evaluate(ori_data[mask])
                if subgroup_predictive_score is not None:
                    subgroup_name = "_".join([f"{int(v)}" for k, v in row.items()])
                    predictive_score[subgroup_name] = subgroup_predictive_score
        return predictive_score
    
    
    