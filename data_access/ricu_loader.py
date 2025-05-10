import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from absl import flags, app
from data_access.base_loader import BaseLoader
from data_access.preprocessing import Standardize, MissingnessDeltaT
import math
from sklearn.model_selection import train_test_split



class RicuLoader(BaseLoader):
    def __init__(self, seed, task_name, data_name,static_var,ricu_dataset_path,simple_imputation = False,
                 features = None, processed_output_path = '',intermed_output_path = '', LOS_cutoff = 3):
        super().__init__(seed, task_name,data_name, static_var, simple_imputation,
                        processed_output_path,intermed_output_path)
        # self.data_path = data_path
        # self.outcomes_path = outcomes_path
        # self.static_path = static_path

        self.ricu_dataset_path = ricu_dataset_path
        self.data_name = data_name
        self.task_name = task_name
        self.LOS_cutoff = LOS_cutoff
        self.dyn = pd.read_parquet(os.path.join(self.ricu_dataset_path,'dyn.parquet'), engine='pyarrow')
        self.outcome = pd.read_parquet(os.path.join(self.ricu_dataset_path,'outc.parquet'), engine='pyarrow')
        self.static = pd.read_parquet(os.path.join(self.ricu_dataset_path,'sta.parquet'), engine='pyarrow')
        self.len_patients = self.dyn['stay_id'].nunique()
        self.time_steps = int(self.dyn['time'].max())
        self.imputation = simple_imputation
        self.seed = seed
        self.features = features
        
    @staticmethod
    def _simple_imputer_tr(df_out):
        ID_COL = 'stay_id'
        df_out = df_out.dropna(axis=1, how='all')
        icustay_means = df_out.groupby(ID_COL).mean()
        global_means = df_out.mean(axis=0)
        # df_out_fill = df_out.groupby(ID_COL).fillna(
        #     method='ffill').groupby(ID_COL).fillna(icustay_means).fillna(global_means)
        
        id_col = df_out[ID_COL]

        # Forward fill missing values within each group
        df_out_fill = df_out.groupby(id_col).ffill()

        # Fill remaining missing values with group means
        df_out_fill = df_out_fill.fillna(icustay_means)

        # Fill any remaining NaNs with global means
        df_out_fill = df_out_fill.fillna(global_means)

        # Add the ID column back to the DataFrame
        df_out_fill[ID_COL] = id_col

        return df_out_fill
    
    
    def _reshape_vitals(self, vitals_np):
        return np.transpose(
            np.reshape(vitals_np, (self.len_patients, self.time_steps + 1, -1)),
            (0, 2, 1)
        )
    def get_timeseries(self):
        """
        Returns array containing raw time series per feature per patient.

        Returns:
            vitals_out: Array of all patients' vitals timeseries [N_patients, features, time_len]
            feature names: Array of all feature names [features]
        """
        # Load data        
        # read and transform the origianl data into arrays
        temp_original = self.dyn.drop(['stay_id','time'], axis=1)
        vitals_out_original = self._reshape_vitals(temp_original.to_numpy())
        feature_names = np.asarray(temp_original.columns) 
                
        if self.imputation:
            temp_imputed_df = self._simple_imputer_tr(self.dyn)
            temp_imputed_df = temp_imputed_df.drop(['stay_id','time'], axis=1)
            vitals_out_imputed = self._reshape_vitals(temp_imputed_df.to_numpy())

        else:
            vitals_out_imputed = np.nan_to_num(vitals_out_original)
        
        patients_intermed = self.get_intermed_preprocessors().transform(vitals_out_imputed, vitals_out_original)
        patients_intermed['feature_names'] = feature_names
        
        return patients_intermed, feature_names


    def get_outcomes(self):
        """
        Returns array containing labels time series per target per patient.

        Returns:
            outcomes_out: Array of all patients' labels [N_patients, targets]
        """
        # Load data
        temp = self.outcome.drop(['stay_id'], axis=1)


        if self.task_name == 'los_24':
            temp['label']= temp['label']/24
            temp['label'] = (temp['label'] > self.LOS_cutoff).astype(int)

        outcomes_np = temp.to_numpy()

        return outcomes_np


    def get_static_data(self):
        """
        Returns array containing static data for each patient.

        Returns:
            static_data: [N_patients, static_features]
            feature_names: [static_features]
        """
        # Load data
        # static_raw = pd.read_csv(self.static_path)
        icu_ids = self.static['stay_id'].to_numpy()
        temp = self.static.drop(['stay_id'], axis=1)
        temp = temp.rename(columns={'sex': 'gender'})
        temp = temp.rename(columns={'ethnic': 'ethnicity'})

        # Map features to integer encoding
        def ethnicity_map(item):
            if 'white' in item.lower():
                return 0
            elif 'black' in item.lower():
                return 1
            elif 'asian' in item.lower():
                return 2
            # elif 'hispanic' in item.lower():
            #     return 3
            else:
                return 3

        # def insurance_map(item):
        #     if 'Medicare' in item:
        #         return 0
        #     elif 'Medicaid' in item:
        #         return 1
        #     elif 'Private' in item:
        #         return 2
        #     elif 'Government' in item:
        #         return 3
        #     elif 'Self' in item:
        #         return 4
        #     else:
        #         raise ValueError

        # def admission_type_map(item):
        #     if 'EMERGENCY' in item:
        #         return 0
        #     elif 'ELECTIVE' in item:
        #         return 1
        #     elif 'URGENT' in item:
        #         return 2
        #     else:
        #         raise ValueError

        # def first_careunit_map(item):
        #     if 'MICU' in item:
        #         return 0
        #     elif 'CSRU' in item:
        #         return 1
        #     elif 'SICU' in item:
        #         return 2
        #     elif 'CCU' in item:
        #         return 3
        #     elif 'TSICU' in item:
        #         return 4
        #     else:
        #         raise ValueError


        temp['gender'] = [0 if i == 'Female' else 1 for i in temp['gender']]
        # temp['ethnicity'].fillna('unknown', inplace=True)
        temp['ethnicity'] = temp['ethnicity'].fillna('unknown')
        temp['ethnicity'] = list(map(ethnicity_map, temp['ethnicity']))
 
        
        # Handling age
        temp['age'] = [int(i) if i <= 90 else 90 for i in temp['age']]
        age_bins = [0, 30, 50, 70, 100]
        temp['age_group'] = pd.cut(temp['age'], bins=age_bins, labels = False)

        age_intervals = pd.cut(temp['age'], bins=age_bins).cat.categories
        age_interval_strings = [str(interval) for interval in age_intervals]
        age_interval_mapping = {interval: idx for idx, interval in enumerate(age_interval_strings)}
        print('Mapped age groups:')
        print(age_interval_mapping)
        
        
        # Handling BMI
        temp['bmi'] = 10000*temp['weight'] / (temp['height'] * temp['height'])
        bmi_bins = [0, 18.5, 24.9, 29.9, 100]
        temp['bmi_group'] = pd.cut(temp['bmi'], bins=bmi_bins,labels = False)
        bmi_intervals = pd.cut(temp['bmi'], bins=bmi_bins).cat.categories
        bmi_interval_strings = [str(interval) for interval in bmi_intervals]
        bmi_interval_mapping = {interval: idx for idx, interval in enumerate(bmi_interval_strings)}
        print('Mapped bmi groups:')
        print(bmi_interval_mapping)

        
        static_out = {'c': temp.to_numpy(),
                    'feature_names': np.asarray(temp.columns),
                    'ids':icu_ids}

        return static_out


    def select_features(self, X, features):
        """
        Args:
            X: dict containing at least 'inputs': [N_patients, features, time_len]
               and 'feature_names': [features]
            features: list of features to keep

        Returns:
            X: dict with entries containing only the desired features
        """
        if features is None:
            return X
        else:
            # feature_idxs_to_keep = np.where(np.isin(X['feature_names'], features)).tolist()
            feature_mask = np.isin(X['feature_names'], features)
            X_tf = {}
            for key, value in X.items():
                try:
                    X_tf[key] = value[:,feature_mask,...]
                except IndexError: # Handle feature_names
                    X_tf[key] = value[feature_mask]
            return X_tf
    
    def select_static_features(self, static, static_var):
        """
        Returns associated labels of the specified task.

        Args:
            outcomes: all available static features
            task: name of the chosen task
        """
        
        if static_var is None:
            return static
        else:
            feature_names = static['feature_names']
            idx = []
            for var in static_var:
                idx.append(np.where(feature_names == var)[0][0])

            return static['c'][:,idx],idx

    def select_task(self, outcomes, task):
        """
        Returns associated labels of the specified task.

        Args:
            outcomes: all available outcomes
            task: name of the chosen task
        """
        return outcomes


    def get_intermed_preprocessors(self):

        preprocessor = MissingnessDeltaT()

        return preprocessor

    def get_input_preprocessors(self):
        preprocessors = [
            Standardize(),
        ]

        return preprocessors
    def get_intersectional_indexes(self, static, type, train_fraction, val_fraction, oracle_fraction, oracle_min, 
                                 intersectional_min_threshold, intersectional_max_threshold):
        """
        Splits patient data into different sets based on demographic intersections while preserving class distributions.
        
        Data Splitting Strategy:
        -----------------------
        1. Group Classification:
           - Majority Groups: Intersections with > intersectional_max_threshold samples
           - Minority Groups: Intersections with samples between intersectional_min_threshold and intersectional_max_threshold
           - Excluded Groups: Intersections with < intersectional_min_threshold samples (not used)
        
        2. Splitting Strategy:
           
           Majority Groups - Split into three sets:
           - Training Set: train_fraction of samples
           - Validation Set: val_fraction of samples
           - Test Set: remaining samples (1 - train_fraction - val_fraction)
           
           Minority Groups - Split into four sets:
           
           Normal Case (group size > 1.3 × oracle_min):
           - Oracle Set: max(oracle_fraction × total_size, oracle_min)
           - Remaining samples split into:
             * Training Set: train_fraction of remaining
             * Validation Set: val_fraction of remaining
             * Test Set: remaining samples
           
           Edge Case (group size ≤ 1.3 × oracle_min):
           - Oracle Set (exactly oracle_min samples):
             * Positive class: Maximum 60% of available positive samples
             * Negative class: Remaining slots filled with negative samples
           - Remaining samples split according to train_fraction and val_fraction
        
        3. Class Balance:
           - All splits are performed separately for each class (positive/negative outcomes)
           - Original class distribution is maintained within each demographic intersection
           - For edge cases, positive samples (minority) are deliberately preserved in non-oracle sets
        
        Args:
            static (pd.DataFrame): DataFrame containing demographic and outcome data
            type (str): Either 'majority' or 'minority' to indicate group type
            train_fraction (float): Fraction of data for training set
            val_fraction (float): Fraction of data for validation set
            oracle_fraction (float): Target fraction for oracle set (minority groups only)
            oracle_min (int): Minimum size for oracle set (minority groups only)
            intersectional_min_threshold (int): Minimum size for a group to be included
            intersectional_max_threshold (int): Threshold between majority and minority groups
        
        Returns:
            If type == 'majority':
                train_indices, val_indices, test_indices
            If type == 'minority':
                train_indices, val_indices, test_indices, oracle_indices
        """
        def _shuffle_arrays_in_unison(array1, array2):
            assert len(array1) == len(array2), "Both arrays must have the same length"
            p = np.random.permutation(len(array1))
            return array1[p], array2[p]    

        def _check_can_stratify(labels, min_samples_per_class=2, next_split_size=None):
            """
            Check if current set of labels can be stratified considering the next split size.
            
            Args:
                labels: Array of labels
                min_samples_per_class: Minimum samples needed per class (default: 2)
                next_split_size: Size of the next split (if None, only checks current distribution)
            """
            unique, counts = np.unique(labels, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            print(f"Label distribution: {counts_dict}")
            
            # First check if we have minimum samples per class
            has_min_samples = all(count >= min_samples_per_class for count in counts)
            
            # If we need to consider next split, check if each class will have enough samples
            if next_split_size is not None and has_min_samples:
                n_total = len(labels)
                split_ratio = next_split_size / n_total
                # Check if each class will have enough samples in the next split
                has_min_samples = all(count * split_ratio >= min_samples_per_class for count in counts)
                if not has_min_samples:
                    print(f"Warning: Next split of size {next_split_size} would have insufficient samples per class")
                    print(f"Estimated next split distribution: {dict(zip(unique, counts * split_ratio))}")
            
            return has_min_samples

        def _stratified_array_split(points, labels, type, train_fraction, val_fraction, oracle_fraction, oracle_min):
            if type == 'majority':
                # Majority split logic remains the same
                size = len(points)
                train_size = math.floor(train_fraction * size)
                val_size = math.ceil(val_fraction * size)
                test_size = size - train_size - val_size
                
                print(f"\nTarget sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
                
                # Calculate proportions
                total_prop = train_size + val_size + test_size
                train_prop = train_size / total_prop
                val_prop = val_size / total_prop
                test_prop = test_size / total_prop
                
                train_points, val_points, test_points = [], [], []
                unique_labels = np.unique(labels)
                
                # Split each class according to proportions
                for label in unique_labels:
                    label_mask = labels == label
                    label_points = points[label_mask]
                    n_samples = len(label_points)
                    
                    # Calculate sizes for this class
                    n_train = int(np.round(n_samples * train_prop))
                    n_val = int(np.round(n_samples * val_prop))
                    n_test = n_samples - n_train - n_val  # Ensure we use all samples
                    
                    print(f"Class {label} distribution - Total: {n_samples}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
                    
                    # Shuffle points for this class
                    np.random.seed(42)
                    shuffled_points = np.random.permutation(label_points)
                    
                    # Split points
                    start_idx = 0
                    train_points.extend(shuffled_points[start_idx:start_idx + n_train])
                    start_idx += n_train
                    
                    val_points.extend(shuffled_points[start_idx:start_idx + n_val])
                    start_idx += n_val
                    
                    test_points.extend(shuffled_points[start_idx:])
                
                return train_points, val_points, test_points
            
            else:
                # Minority split logic
                if len(points) <= oracle_min:
                    return [], [], [], []
                    
                unique_labels = np.unique(labels)
                train_points, val_points, test_points, oracle_points = [], [], [], []
                
                # Get counts per class
                label_counts = {label: np.sum(labels == label) for label in unique_labels}
                total_size = len(points)
                
                # Special handling for cases close to oracle_min
                is_edge_case = total_size < oracle_min * 1.3  # Within 30% of oracle_min
                
                if is_edge_case:
                    print(f"\nEdge case detected: total size {total_size} close to oracle_min {oracle_min}")
                    # For edge cases, we fix oracle size to exactly oracle_min
                    oracle_size = oracle_min
                    remaining_size = total_size - oracle_min
                    
                    # Calculate target proportions for oracle set
                    # We want to keep more positives in non-oracle sets
                    pos_label = max(unique_labels)  # Assuming binary classification with 1 as positive
                    n_pos = label_counts[pos_label]
                    n_neg = total_size - n_pos
                    
                    # Allocate at most 60% of positives to oracle
                    max_pos_oracle = int(0.6 * n_pos)
                    # Calculate how many negatives needed to reach oracle_min
                    n_neg_oracle = oracle_min - max_pos_oracle
                    
                    # Distribute remaining samples
                    remaining_pos = n_pos - max_pos_oracle
                    remaining_neg = n_neg - n_neg_oracle
                    
                    # Calculate proportions for remaining sets
                    train_size = int(train_fraction * remaining_size)
                    val_size = int(val_fraction * remaining_size)
                    test_size = remaining_size - train_size - val_size
                    
                    for label in unique_labels:
                        label_mask = labels == label
                        label_points = points[label_mask]
                        n_samples = len(label_points)
                        
                        # Shuffle points for this class
                        np.random.seed(42)
                        shuffled_points = np.random.permutation(label_points)
                        
                        if label == pos_label:
                            n_oracle = max_pos_oracle
                            n_remaining = remaining_pos
                        else:
                            n_oracle = n_neg_oracle
                            n_remaining = remaining_neg
                        
                        # Calculate sizes for non-oracle sets
                        n_train = int(np.round(n_remaining * train_size/remaining_size))
                        n_val = int(np.round(n_remaining * val_size/remaining_size))
                        n_test = n_remaining - n_train - n_val
                        
                        print(f"Class {label} distribution - Total: {n_samples}, Oracle: {n_oracle}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
                        
                        # Split points
                        start_idx = 0
                        oracle_points.extend(shuffled_points[start_idx:start_idx + n_oracle])
                        start_idx += n_oracle
                        
                        train_points.extend(shuffled_points[start_idx:start_idx + n_train])
                        start_idx += n_train
                        
                        val_points.extend(shuffled_points[start_idx:start_idx + n_val])
                        start_idx += n_val
                        
                        test_points.extend(shuffled_points[start_idx:])
                
                else:
                    # Normal case - use original proportional logic
                    oracle_size = max(int(oracle_fraction * total_size), oracle_min)
                    remaining_size = total_size - oracle_size
                    train_size = int(train_fraction * remaining_size)
                    val_size = int(val_fraction * remaining_size)
                    test_size = remaining_size - train_size - val_size
                    
                    print(f"\nTarget sizes - Oracle: {oracle_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
                    
                    # Calculate proportions
                    total_prop = oracle_size + train_size + val_size + test_size
                    oracle_prop = oracle_size / total_prop
                    train_prop = train_size / total_prop
                    val_prop = val_size / total_prop
                    test_prop = test_size / total_prop
                    
                    # Split each class according to proportions
                    for label in unique_labels:
                        label_mask = labels == label
                        label_points = points[label_mask]
                        n_samples = len(label_points)
                        
                        # Calculate sizes for this class
                        n_oracle = int(np.round(n_samples * oracle_prop))
                        n_train = int(np.round(n_samples * train_prop))
                        n_val = int(np.round(n_samples * val_prop))
                        n_test = n_samples - n_oracle - n_train - n_val
                        
                        print(f"Class {label} distribution - Total: {n_samples}, Oracle: {n_oracle}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
                        
                        # Shuffle points for this class
                        np.random.seed(42)
                        shuffled_points = np.random.permutation(label_points)
                        
                        # Split points
                        start_idx = 0
                        oracle_points.extend(shuffled_points[start_idx:start_idx + n_oracle])
                        start_idx += n_oracle
                        
                        train_points.extend(shuffled_points[start_idx:start_idx + n_train])
                        start_idx += n_train
                        
                        val_points.extend(shuffled_points[start_idx:start_idx + n_val])
                        start_idx += n_val
                        
                        test_points.extend(shuffled_points[start_idx:])
                
                return train_points, val_points, test_points, oracle_points

        # Create contingency table
        contingency_table = pd.crosstab(
            index=[static['age_group'], static['ethnicity']], 
            columns=[static['gender'], static['bmi_group']]
        )
        
        # Mask based on type
        if type == 'majority':
            masked_contingency_table = contingency_table.map(
                lambda x: x if (x > intersectional_max_threshold or 
                              (x < intersectional_min_threshold and x > 0)) else pd.NA
            )
        else:
            masked_contingency_table = contingency_table.map(
                lambda x: x if (x < intersectional_max_threshold and 
                              x > intersectional_min_threshold) else pd.NA
            )

        train_indices = []
        val_indices = []
        test_indices = []
        oracle_indices = []
        
        # Process each demographic combination
        for age_group, ethnicity in masked_contingency_table.index:
            for gender in masked_contingency_table.columns.levels[0]:
                for bmi in masked_contingency_table.columns.levels[1]:
                    count = masked_contingency_table.loc[(age_group, ethnicity), (gender, bmi)]
                    if pd.notna(count):
                        # Get indices and labels for current combination
                        subset = static[(static['age_group'] == age_group) & 
                                      (static['ethnicity'] == ethnicity) & 
                                      (static['bmi_group'] == bmi) & 
                                      (static['gender'] == gender)]
                        indices = np.array(subset.index)
                        labels = subset['outcome'].values
                        
                        print(f"\nProcessing group: age={age_group}, ethnicity={ethnicity}, gender={gender}, bmi={bmi}")
                        
                        # Shuffle while keeping pairs together
                        indices, labels = _shuffle_arrays_in_unison(indices, labels)
                        
                        # Split with stratification
                        if type == 'majority':
                            train, val, test = _stratified_array_split(
                                indices, labels, type, train_fraction, val_fraction,
                                oracle_fraction, oracle_min
                            )
                            train_indices.extend(train)
                            val_indices.extend(val)
                            test_indices.extend(test)
                        else:
                            train, val, test, oracle = _stratified_array_split(
                                indices, labels, type, train_fraction, val_fraction,
                                oracle_fraction, oracle_min
                            )
                            train_indices.extend(train)
                            val_indices.extend(val)
                            test_indices.extend(test)
                            oracle_indices.extend(oracle)
        
        if type == 'majority':
            return train_indices, val_indices, test_indices
        else:
            return train_indices, val_indices, test_indices, oracle_indices


