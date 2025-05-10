## Necessary Packages
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import os
# Plotting
import plotly.graph_objects as go
import plotly.express as px
# Suppress specific warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def display_scores(results):
  mean = np.mean(results)
  sigma = scipy.stats.sem(results)
  sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., 5-1)
  #  sigma = 1.96*(np.std(results)/np.sqrt(len(results)))
  mean = round(mean, 4)
  sigma = round(sigma, 4)
  print('Final Score: ', f'{mean} \xB1 {sigma}')
  return mean, sigma
def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len




def visualization(ori_data, generated_data, analysis, use_plotly = True, compare=3000, label = None):
    """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]

    # Data preprocessing
    # ori_data = np.asarray(ori_data)
    # generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)



        if use_plotly:
            # Create a Plotly figure
            fig = go.Figure()

            # Get unique labels and their colors
            colors = px.colors.qualitative.Plotly  # Using Plotly's qualitative color scale

            fig.add_trace(go.Scatter(
          x=pca_results[:, 0], 
          y=pca_results[:, 1],  
          mode='markers',
          name='Original',
          marker=dict(color=colors[0]),
          text=['Original'] * anal_sample_no  # Hover text
            ))
            fig.add_trace(go.Scatter(
          x=pca_hat_results[:, 0], 
          y=pca_hat_results[:, 1],  
          mode='markers',
          name='Synthetic',
          marker=dict(color=colors[1]),
          text=['Synthetic'] * anal_sample_no  # Hover text
            ))
            fig.update_layout(
          title='PCA plot',
          xaxis_title='x-pca',
          yaxis_title='y_pca',
          width=960,  # Width in pixels
          height=768,
          paper_bgcolor='white',  # Sets the background color of the paper to white
          plot_bgcolor='white',  # Sets the background color of the plot area to white
          xaxis=dict(showline=True, linecolor='black', linewidth=2),
          yaxis=dict(showline=True, linecolor='black', linewidth=2)
            )

            # Show plot
            fig.show()
        else:
            # Matplotlib figure
            f, ax = plt.subplots(1)
            plt.scatter(pca_results[:, 0], pca_results[:, 1], c='red', alpha=0.2, label="Original")
            plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c='blue', alpha=0.2, label="Synthetic")
            ax.legend()
            if label == 1:
              plt.title('PCA plot of positive samples')
            elif label == 0:
              plt.title('PCA plot of negative samples')
            elif label is None:
              plt.title('PCA plot')
            plt.xlabel('x-pca')
            plt.ylabel('y_pca')
            plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        if label == 1:
          plt.title('t-SNE plot of positive samples')
        elif label == 0:
          plt.title('t-SNE plot of negative samples')
        elif label is None:
          plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.show()

    elif analysis == 'kernel':
       
        # Visualization parameter
        # colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

        f, ax = plt.subplots(1)
        sns.distplot(prep_data, hist=False, kde=True, kde_kws={'linewidth': 5}, label='Original', color="red")
        sns.distplot(prep_data_hat, hist=False, kde=True, kde_kws={'linewidth': 5, 'linestyle':'--'}, label='Synthetic', color="blue")
        # Plot formatting

        # plt.legend(prop={'size': 22})
        plt.legend()
        plt.xlabel('Data Value')
        plt.ylabel('Data Density Estimate')
        # plt.rcParams['pdf.fonttype'] = 42

        # plt.savefig(str(args.save_dir)+"/"+args.model1+"_histo.png", dpi=100,bbox_inches='tight')
        # plt.ylim((0, 12))
        plt.show()
        plt.close()
def old_evaluate_subgroups(downstream_evaluator, X_data, y_data, static, n_bootstrap,model_path,filename):    
  contingency_table = pd.crosstab(index=[static['age_group'], static['ethnicity']], columns=[static['gender'], static['bmi_group']])

  for age_group, ethnicity in contingency_table.index:
      for gender in contingency_table.columns.levels[0]:
          for bmi_group in contingency_table.columns.levels[1]:
              for split in ['oracle', 'test', 'synthetic']:
                  indices = static[(static['age_group'] == age_group) & 
                                  (static['ethnicity'] == ethnicity) & 
                                  (static['gender'] == gender) & 
                                  (static['bmi_group'] == bmi_group) &
                                  (static['split'] == split)]['index'].to_numpy()
                  
                  X_split = downstream_evaluator.slice_array(X_data[split], indices)
                  y_split = downstream_evaluator.slice_array(y_data[split], indices)
                  
                  if X_split.shape[0] > 0:
                      print(f'evaluating {age_group, ethnicity, gender, bmi_group, split}')
                      dummy = downstream_evaluator.bootstrap_evaluate(X_split, y_split.astype(int), n_bootstrap)
                      dummy.update({'Sample Size': [X_split.shape[0]] * n_bootstrap})
                      conf_intervals = {key: calculate_confidence_interval(value) for key, value in dummy.items()}
                      # print(conf_intervals['auroc'])
                      results_temp = pd.DataFrame.from_dict(conf_intervals, orient='index').reset_index()
                      results_temp.columns = ['key', 'mean', 'conf-interval']
                      results_temp['age_group'] = age_group
                      results_temp['ethnicity'] = ethnicity
                      results_temp['gender'] = gender
                      results_temp['bmi_group'] = bmi_group
                      results_temp['split'] = split
                      results = pd.concat([results, results_temp])
      # results_path = os.path.join(out_path, model_timestamp)
      # Ensure the directory exists
      # os.makedirs(os.path.dirname(results_path), exist_ok=True)
  # Assuming 'mean', 'ci_lower', and 'ci_upper' are column names in your DataFrame 'results'
  results['summary'] = results.apply(lambda row: f"{np.round(row['mean'],3)} {row['conf-interval']}" if row['key']!= 'Sample Size' else row['mean'], axis=1)
  results['subgroup'] = results.apply(lambda row: f"{row['age_group']}_{row['ethnicity']}_{row['gender']}_{row['bmi_group']}" , axis=1)
  results.to_csv(os.path.join(model_path,filename), index=False)
  
def evaluate_subgroups(downstream_evaluator, data_dict,cond_names, n_bootstrap, model_path, filename, verbose = False):
    """
    Evaluate model performance on intersectional subgroups across test, synthetic, and oracle data.
    
    Args:
        downstream_evaluator: Model evaluator instance
        data_dict: Dictionary containing:
            {
                'test': {'x': x_test, 'y': y_test, 'c': c_test},
                'synthetic': {'x': x_synth, 'y': y_synth, 'c': c_synth},
                'oracle': {'x': x_oracle, 'y': y_oracle, 'c': c_oracle}
            }
        n_bootstrap: Number of bootstrap iterations
        model_path: Path to save results
        filename: Name of the output file
        
    Returns:
        pd.DataFrame: Results for each subgroup and data split
    """
    results = pd.DataFrame()
    # demographic_cols = ['age_group', 'ethnicity', 'bmi_group', 'gender']
    demographic_cols = list(cond_names)

    # Get all unique combinations across all splits
    all_combinations = []
    for split, split_data in data_dict.items():
        # Convert the array 'c' into a DataFrame for easier manipulation
        df = pd.DataFrame(split_data['c'], columns=demographic_cols)
        combinations = df[demographic_cols].drop_duplicates()
        combinations['split'] = split
        all_combinations.append(combinations)
    all_combinations = pd.concat(all_combinations).drop_duplicates()
    
    # Evaluate each combination
    for _, row in tqdm(all_combinations.iterrows()):
        split = row['split']
        split_data = data_dict[split]
        
        # Create mask for this demographic combination
        mask = np.ones(len(split_data['c']), dtype=bool)
        for col in demographic_cols:
            mask &= split_data['c'][:, demographic_cols.index(col)] == row[col]
        
        # Get data for this subgroup
        X_subgroup = split_data['x'][mask]
        y_subgroup = split_data['y'][mask]
        
        if len(X_subgroup) > 0:
          if verbose:
            print(f'Evaluating {dict(row)}')
          metrics = downstream_evaluator.bootstrap_evaluate(X_subgroup, y_subgroup.astype(int), n_bootstrap,verbose)
          metrics.update({'Sample Size': [len(X_subgroup)] * n_bootstrap})
          
          # Calculate confidence intervals
          conf_intervals = {key: calculate_confidence_interval(value) for key, value in metrics.items()}
          
          # Create results row
          results_temp = pd.DataFrame.from_dict(conf_intervals, orient='index').reset_index()
          results_temp.columns = ['key', 'mean', 'conf-interval']
          
          # Add demographic information
          for col in demographic_cols:
              results_temp[col] = row[col]
          results_temp['split'] = split
          
          results = pd.concat([results, results_temp])
    
    # Format results
    results['summary'] = results.apply(
        lambda row: f"{np.round(row['mean'],3)} {row['conf-interval']}" if row['key']!= 'Sample Size' else row['mean'], 
        axis=1
    )
    results['subgroup'] = results.apply(
        lambda row: f"{row['age_group']}_{row['ethnicity']}_{row['gender']}_{row['bmi_group']}", 
        axis=1
    )
    results[['CI Lower', 'CI Upper']] = results['conf-interval'].apply(lambda x: pd.Series(x))
    
    # Save results
    os.makedirs(os.path.dirname(os.path.join(model_path, filename)), exist_ok=True)
    results.to_csv(os.path.join(model_path, filename), index=False)
    
    return results



def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given array.
    
    Parameters:
    - data: array-like, the data for which to calculate the confidence interval.
    - confidence: float, the confidence level for the interval.
    
    Returns:
    - mean: float, the mean of the data.
    - conf_interval: tuple of float, the lower and upper bounds of the confidence interval.
    """
    # Filter out null values
    cleaned_data = pd.Series(np.array(data)).dropna().to_numpy()
    
    # Check if there's enough data to calculate a confidence interval
    if len(cleaned_data) < 1:
        return np.nan, (np.nan, np.nan)  # or any other indication that CI can't be calculated
    n = len(cleaned_data)

    mean = np.mean(cleaned_data)
    std_dev = np.std(cleaned_data)
    # sem = st.sem(cleaned_data)  # Standard error of the mean
    margin_of_error = scipy.stats.t.ppf((1 + confidence) / 2, n - 1) * (std_dev / np.sqrt(n))
    lower_bound = np.round(mean - margin_of_error,3)
    upper_bound = np.round(mean + margin_of_error,3)
    return mean, (lower_bound, upper_bound)


def tsne_projection(data, num_classes, color_dict=None, class_dict=None, compare=3000):
   # TSNE anlaysis
   tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
   # Plotting
   f, ax = plt.subplots(1)
   for i in range(num_classes):
      data_class = data[i]
      no, seq_len, dim = data_class.shape
      anal_sample_no = min([compare, data_class.shape[0]])
      idx = np.random.permutation(data_class.shape[0])[:anal_sample_no]
      data_class = data_class[idx]

      for j in range(anal_sample_no):
        if (j == 0):
            prep_data = np.reshape(np.mean(data_class[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(data_class[j, :, :], 1), [1, seq_len])))
            
      colors = [color_dict[i] for j in range(anal_sample_no)]

      tsne_results = tsne.fit_transform(prep_data)
      plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                  c=colors[:], alpha=0.2, label=class_dict[i])
      
   ax.legend()

   plt.title('t-SNE plot')
   plt.xlabel('x-tsne')
   plt.ylabel('y_tsne')
   plt.show()

def find_unique_test_combinations(static_c_all):
    """
    Find rows in test set whose demographic combinations don't exist in oracle set.
    
    Args:
        static_c_all (pd.DataFrame): DataFrame containing demographic information and splits
        
    Returns:
        pd.DataFrame: All rows from test set whose demographic combinations don't appear in oracle set
    """
    # Get demographic columns
    demographic_cols = ['age_group', 'ethnicity', 'bmi_group', 'gender']
    
    # Get unique combinations from oracle set
    oracle_combinations = static_c_all[static_c_all['split'] == 'oracle'][demographic_cols].drop_duplicates()
    
    # Get all test rows
    test_rows = static_c_all[static_c_all['split'] == 'test']
    
    # Find test rows whose combinations don't exist in oracle
    test_only_rows = test_rows.merge(
        oracle_combinations, 
        on=demographic_cols, 
        how='left', 
        indicator=True
    )
    test_only_rows = test_only_rows[test_only_rows['_merge'] == 'left_only'].drop('_merge', axis=1)
    
    return test_only_rows

def find_unique_oracle_combinations(static_c_all):
    """
    Find rows in oracle set whose demographic combinations don't exist in test set.
    
    Args:
        static_c_all (pd.DataFrame): DataFrame containing demographic information and splits
        
    Returns:
        pd.DataFrame: All rows from oracle set whose demographic combinations don't appear in test set
    """
    # Get demographic columns
    demographic_cols = ['age_group', 'ethnicity', 'bmi_group', 'gender']
    
    # Get unique combinations from test set
    test_combinations = static_c_all[static_c_all['split'] == 'test'][demographic_cols].drop_duplicates()
    
    # Get all oracle rows
    oracle_rows = static_c_all[static_c_all['split'] == 'oracle']
    
    # Find oracle rows whose combinations don't exist in test
    oracle_only_rows = oracle_rows.merge(
        test_combinations, 
        on=demographic_cols, 
        how='left', 
        indicator=True
    )
    oracle_only_rows = oracle_only_rows[oracle_only_rows['_merge'] == 'left_only'].drop('_merge', axis=1)
    
    return oracle_only_rows

def compare_group_performance(df, metric='auroc', alpha=0.05):
    """
    Perform statistical testing between groups in evaluation results.
    
    Args:
        df: DataFrame containing evaluation results with columns:
            - target_subgroup (tested_on)
            - eval_random_state
            - trained_on
            - metric values (e.g. auroc)
        metric: str, name of the metric column to compare
        alpha: significance level for statistical tests
        
    Returns:
        DataFrame containing test results with columns:
        - tested_on: target subgroup
        - group1: first training condition
        - group2: second training condition
        - random_state: evaluation random state
        - statistic: test statistic
        - pvalue: p-value from statistical test
        - significant: boolean indicating if difference is significant
    """
    from scipy import stats
    import itertools
    
    results = []
    
    # For each target subgroup and random state
    for subgroup in df['target_subgroup'].unique():
        for state in df['eval_random_state'].unique():
            # Get data for this combination
            subset = df[
                (df['target_subgroup'] == subgroup) & 
                (df['eval_random_state'] == state)
            ]
            
            # Get all pairs of training conditions
            train_conditions = subset['trained_on'].unique()
            pairs = list(itertools.combinations(train_conditions, 2))
            
            # Perform statistical test for each pair
            for cond1, cond2 in pairs:
                scores1 = subset[subset['trained_on'] == cond1][metric].values
                scores2 = subset[subset['trained_on'] == cond2][metric].values
                
                # Use Mann-Whitney U test (non-parametric)
                statistic, pvalue = stats.mannwhitneyu(
                    scores1, scores2, 
                    alternative='two-sided'
                )
                
                results.append({
                    'tested_on': subgroup,
                    'group1': cond1,
                    'group2': cond2,
                    'random_state': state,
                    'statistic': statistic,
                    'pvalue': pvalue,
                    'significant': pvalue < alpha,
                    'mean_diff': scores1.mean() - scores2.mean()
                })
    
    return pd.DataFrame(results)

def summarize_significance_tests(test_results, metric='auroc'):
    """
    Summarize the results of statistical tests across all conditions.
    
    Args:
        test_results: DataFrame output from compare_group_performance
        metric: str, name of the metric being compared
        
    Returns:
        DataFrame with summary statistics and formatted string output
    """
    summary = []
    
    for subgroup in test_results['tested_on'].unique():
        subset = test_results[test_results['tested_on'] == subgroup]
        
        # Count number of significant differences
        n_sig = subset['significant'].sum()
        n_total = len(subset)
        
        # Get pairs with significant differences
        sig_pairs = subset[subset['significant']].apply(
            lambda x: f"{x['group1']} vs {x['group2']} (diff={x['mean_diff']:.3f}, p={x['pvalue']:.3e})", 
            axis=1
        ).tolist()
        
        summary.append({
            'tested_on': subgroup,
            'n_significant': n_sig,
            'n_total_tests': n_total,
            'significant_pairs': sig_pairs
        })
    
    return pd.DataFrame(summary)

if __name__ == '__main__':
   class_dict = {0:'StandingUpFS', 1:'StandingUpFL', 2:'Walking', 3:'Running',
                 4:'GoingUpS', 5:'Jumping', 6:'GoingDownS', 7:'LyingDownFS', 8:'SittingDown'}

   color_dict = {0:'lightblue', 1:'lightcoral', 2:'lightcyan', 3:'lightgoldenrodyellow',
                 4:'lightgreen', 5:'lightgray', 6:'lightpink', 7:'lightsalmon', 8:'lightseagreen'}

   num_classes = 9
   ori_data = []
   fake_data = []

   for i in range(num_classes):
       ori_data_i = np.load(f'./OUTPUT/samples/activity_norm_{i}_truth.npy')
       ori_data_i = (ori_data_i + 1) / 2
       ori_data.append(ori_data_i)

       fake_data_i = np.load(f'./OUTPUT/samples/ddpm_fake_activity_{i}.npy')
       fake_data_i = (fake_data_i + 1) / 2
       fake_data.append(fake_data_i)
    
   tsne_projection(ori_data, num_classes, color_dict=color_dict, class_dict=class_dict, compare=12000)