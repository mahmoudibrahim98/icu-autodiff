import six
import numpy as np
# import healthgen.apps.global_parameters
from abc import ABCMeta, abstractmethod
from absl import flags
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import os 
from sklearn.model_selection import StratifiedShuffleSplit
import math
from datetime import datetime
import json
import warnings
# FLAGS = flags.FLAGS
@six.add_metaclass(ABCMeta)
class BaseLoader(object):
    def __init__(self, seed, task_name, data_name,static_var,simple_imputation,
                processed_output_path = '', intermed_output_path = ''):
        self.randomstate = np.random.RandomState(seed)
        self.seed = seed
        self.task_name = task_name
        self.data_name = data_name
        self.static_var = static_var
        # self.extracted_intermed_features_path = extracted_intermed_features_path
        # self.extracted_intermed_labels_path = extracted_intermed_labels_path
        # self.extracted_intermed_static_path = extracted_intermed_static_path

        # self.processed_features_path = processed_features_path
        # self.processed_labels_path = processed_labels_path
        # self.processed_static_path = processed_static_path
        self.imputation = simple_imputation
        self.processed_output_path = processed_output_path
        self.intermed_output_path = intermed_output_path
        # self.save_intermed_data = save_intermed_data
        # self.save_processed_data = save_processed_data

        # self.oracle_fraction = oracle_fraction
        # self.test_fraction = test_fraction
        # self.val_fraction = val_fraction
        # self.train_fraction = train_fraction    

    @abstractmethod
    def get_intersectional_indexes(self, static, type, train_fraction, val_fraction, oracle_fraction):
        
        raise NotImplementedError


    @abstractmethod
    def get_timeseries(self):
        """
        Loads raw data from some source and returns formatted patient time series.
        Returns:
            patients_ts: [N_patients, features, time_len]
            feature_names: [features]
        """
        raise NotImplementedError

    @abstractmethod
    def get_outcomes(self):
        """
        Loads the labels for various outcomes/prediction tasks.
        Returns:
            outcomes: [N_patients, outcomes]
        """
        raise NotImplementedError

    @abstractmethod
    def get_static_data(self):
        """
        Loads static data for patients.
        Returns:
            static_data: [N_patients, static_data]
            feature_names: [static_data]
        """
        raise NotImplementedError

    @abstractmethod
    def select_task(self, outcomes, task_name):
        """
        Select task and return respective labels.
        """
        raise NotImplementedError

    @abstractmethod
    def get_intermed_preprocessors(self):
        raise NotImplementedError

    @abstractmethod
    def get_input_preprocessors(self):
        raise NotImplementedError

    @abstractmethod
    def select_features(self, X, features):
        raise NotImplementedError
    
    @abstractmethod
    def select_static_features(self, X, features):
        raise NotImplementedError

    # @abstractmethod
    # def save_intermed(self, X, y, extracted_intermed_features_path,extracted_intermed_labels_path,extracted_intermed_static_path):
    #     raise NotImplementedError

    # @abstractmethod
    # def save_processed(self, X, y, out_path):
    #     raise NotImplementedError

    # @staticmethod
    # def _train_val_test_split(X, y):
    @staticmethod
    def single_sample_subgroups_mask(demographic_data,):
        """
        Remove subgroups with only one sample from the demographic data and other arrays.

        Parameters:
        - demographic_data: np.ndarray
            The main demographic data array.
        - other_arrays: np.ndarray
            Other arrays with the same structure to filter.

        Returns:
        - filtered_demographic_data: np.ndarray
            The filtered demographic data.
        - filtered_other_arrays: list of np.ndarray
            List of filtered other arrays.
        """
        # Find unique rows and their indices
        unique_rows, indices, counts = np.unique(demographic_data, axis=0, return_index=True, return_counts=True)

        # Get the indices of unique rows that have only one sample
        single_sample_indices = indices[counts == 1]

        # Create a mask to keep rows that are not in single_sample_indices
        mask = np.isin(np.arange(demographic_data.shape[0]), single_sample_indices, invert=True)

        # Filter the demographic data
        # filtered_demographic_data = demographic_data[mask]


        return mask
    def split_dataset(self, X_in, y, c, val_fraction, test_fraction, demographics_to_stratify_on = ['age_group','ethnicity','gender']):
        """
        Splits dataset into stratified (based on the outcome and  demographics_to_stratify_on) train/val/test.

        Args:
            X_in: dict with where entries are asuumed to share the first dimension.
               Includes at least 'X' and 'feature_names' keys.

        Returns:
            X_dict: dictionary with X_train, X_val, X_test
            y_dict: dictionary with y_train, y_val, y_test
        """
        key_list = []
        input_list = []
        for key, value in X_in.items():
            if key != 'feature_names':
                key_list.append(key)
                input_list.append(value)

        # c_dict = {}
        # for key, value in c.items():
        #     c.update({key: value})

                
        X = np.stack(input_list, axis=1)
        train_fraction = 1.0 - val_fraction - test_fraction

        # y = np.concatenate((np.expand_dims(y,1), c['c']), axis=1)
        y = np.concatenate((y, c['c']), axis=1)

        if len(y.shape) == 1: # stratify only if we have a single task
            y_strat = y
        else:
            cond_indices = [list(c['feature_names']).index(name)+1 for name in demographics_to_stratify_on]
            cond_indices.append(0) # append the outcome index
            y_strat = np.array(['_'.join(map(lambda x: str(int(x)), row)) for row in y[:, cond_indices]])

        mask = BaseLoader.single_sample_subgroups_mask(y_strat)
        X_train, X_intermed, y_train, y_intermed = train_test_split(X[mask], y[mask],
                                                                    test_size=1 - train_fraction,
                                                                    random_state=self.randomstate,
                                                                    stratify=y_strat[mask])

        if len(y_intermed.shape) == 1: # stratify only if we have a single task
            y_intermed_strat = y_intermed
        else:
            cond_indices = [list(c['feature_names']).index(name)+1 for name in demographics_to_stratify_on]
            cond_indices.append(0) # append the outcome index
            y_intermed_strat = np.array(['_'.join(map(lambda x: str(int(x)), row)) for row in y_intermed[:, cond_indices]])
        mask_intermed = BaseLoader.single_sample_subgroups_mask(y_intermed_strat)
        X_val, X_test, y_val, y_test = train_test_split(X_intermed[mask_intermed], y_intermed[mask_intermed],
                                                        test_size=test_fraction / (test_fraction + val_fraction),
                                                        random_state=self.randomstate,
                                                        stratify=y_intermed_strat[mask_intermed])

        X_dict = {}
        for idx, key in enumerate(key_list):
            X_dict[F'{key}_train'] = X_train[:,idx,...]
            X_dict[F'{key}_val'] = X_val[:, idx, ...]
            X_dict[F'{key}_test'] = X_test[:, idx, ...]
        X_dict['feature_names'] = X_in['feature_names']

        # The features taken for y are the first column of the y array (first intervention corresponding to ''vent'')
        # while the features taken for c are all the columns of the y array except the first column (containing the 13 
        # other interventions, 12 window labels, and 12 static features)
        y_dict = {
            'y_train': y_train[:,0],
            'y_val': y_val[:,0],
            'y_test': y_test[:,0]
        }

        c_dict = {
            'c_train': y_train[:,1:],
            'c_val': y_val[:,1:],
            'c_test': y_test[:,1:],
            'feature_names': c['feature_names']
        }

        y_dict.update(c_dict)

        return X_dict, y_dict

    @staticmethod
    def _select_array(arr, values):
        """
        Selects elements from an array based on a given value.

        Parameters:
        arr (numpy.ndarray): The input array.
        val: The value to select elements based on.

        Returns:
        numpy.ndarray: The selected elements from the array.
        numpy.ndarray: The indexes of the selected elements.
        """
        # Find the indexes where the array is equal to the given value
        indexes = np.where(np.isin(arr, values))[0]
        # Return the indexes of the selected elements
        return  indexes
    
    def label_static_stratify(self,y,static_var_idx):
        label_only = y[:,0]
        all_c = y[:,1:]
        relevent_c=all_c[:,static_var_idx]
        relevent_y = np.concatenate((np.expand_dims(label_only,1), np.expand_dims(relevent_c,1)), axis=1)
        relevent_y_str = relevent_y.astype(str)
        relevent_y_strat = np.apply_along_axis(''.join, 1, relevent_y_str)
        return relevent_y_strat
    @staticmethod
    def find_combinations(m, increment=0.1):
        combinations = []
        current_value = increment

        while current_value < m:
            a = round(current_value, 1)
            b = round(m - a, 1)
            if a != 0.0 and b != 0.0:
                combinations.append((a, b))
            current_value += increment
        
        return combinations




    def majority_minority_split(self, X, c_in, label, static_var, train_fraction, val_fraction, increment=0.1):
        """
        Split the input data into training, validation, testing, and oracle sets based on majority and minority labels.

        Parameters:
        - X (numpy.ndarray):       The input data array.
        - c (numpy.ndarray):       The static labels array.
        - label (numpy.ndarray):   The labels array.
        - train_fraction (float):  The fraction of the data used for training.
        - val_fraction (float):    The fraction of the data used for validation.
        - majority_labels (list):  The labels considered as majority. Default is [0].
        - minority_labels (list):  The labels considered as minority. Default is [1, 2, 3, 4].
        - increment (float):       The increment step for generating combinations. Default is 0.1.

        Returns:
        - List of tuples containing:
        - X_train (numpy.ndarray): The training data array.
        - X_val (numpy.ndarray): The validation data array.
        - X_test (numpy.ndarray): The testing data array.
        - X_oracle (numpy.ndarray): The oracle data array.
        - y_train (numpy.ndarray): The training labels array.
        - y_val (numpy.ndarray): The validation labels array.
        - y_test (numpy.ndarray): The testing labels array.
        - y_oracle (numpy.ndarray): The oracle labels array.
        - patients_train (list): The training patients list.
        - patients_val (list): The validation patients list.
        - patients_test (list): The testing patients list.
        - patients_oracle (list): The oracle patients list.
        """
        results = {}
        remaining_fraction = round(1.0 - (train_fraction + val_fraction),1)
        assert remaining_fraction > 0, "The sum of train_fraction and val_fraction must be less than 1.0"

        # Generate oracle-test combinations
        oracle_test_combinations = BaseLoader.find_combinations(remaining_fraction, increment)

        c, static_var_idx = self.select_static_features(c_in, static_var)  # Select relevant static variable

        patient_ids = c_in['ids']
        c_in = c_in['c']
        
        y = np.concatenate((label.reshape(-1, 1), c_in), axis=1)

        # Create stratification keys
        y_strat = np.core.defchararray.add(label.astype(str), c.astype(str))

        # Split off the training and validation sets
        X_train_val, X_remaining, y_train_val, y_remaining, patients_train_val, patients_remaining = train_test_split(
            X, y, patient_ids, test_size=remaining_fraction, random_state=self.randomstate, stratify=y_strat
        )

        y_train_val_strat = self.label_static_stratify(y_train_val,static_var_idx)


        # Split training and validation from the combined set
        X_train, X_val, y_train, y_val, patients_train, patients_val = train_test_split(
            X_train_val, y_train_val, patients_train_val, test_size=val_fraction / (train_fraction + val_fraction), random_state=self.randomstate, stratify=y_train_val_strat
        )

        y_remaining_strat = self.label_static_stratify(y_remaining,static_var_idx)

        results.update({'train': (X_train, y_train), 'val': (X_val, y_val), 'test' :[], 'oracle': []})  
        for test_fraction, oracle_fraction in oracle_test_combinations:
            adjusted_test_fraction = (test_fraction / (test_fraction + oracle_fraction)) 
            adjusted_oracle_fraction = (oracle_fraction / (test_fraction + oracle_fraction)) 

            X_test, X_oracle, y_test, y_oracle, patients_test, patients_oracle = train_test_split(
                X_remaining, y_remaining, patients_remaining, test_size=adjusted_oracle_fraction, random_state=self.randomstate, stratify=y_remaining_strat
            )

            results['test'].append(( X_test,  y_test))
            results['oracle'].append(( X_oracle, y_oracle))

        return results
    

            


        
    def intersectional_split(self, X, c_in, label, train_fraction, val_fraction, oracle_fraction,oracle_min,intersectional_min_threshold,intersectional_max_threshold):
        static_df = pd.DataFrame(c_in['c'], columns=c_in['feature_names'])
        label_df = pd.DataFrame(label,columns = ['outcome'])
        labels_and_static = np.concatenate((label.reshape(-1,1), c_in['c']), axis=1)

        combined_df = pd.concat([static_df,label_df],axis=1)
        # age_bins = [0, 30, 50, 70, 100]
        # combined_df['age_group'] = pd.cut(combined_df['age'], bins=age_bins)
        
        train_indeces_majority, val_indeces_majority, test_indeces_majority = self.get_intersectional_indexes(combined_df, 'majority', train_fraction, val_fraction,oracle_fraction,oracle_min,intersectional_min_threshold,intersectional_max_threshold)
        train_indeces_minority, val_indeces_minority, test_indeces_minority, oracle_indeces_minority = self.get_intersectional_indexes(combined_df, 'minority', train_fraction, val_fraction, oracle_fraction,oracle_min,intersectional_min_threshold,intersectional_max_threshold)
        
        train_indeces = train_indeces_majority + train_indeces_minority
        val_indeces = val_indeces_majority + val_indeces_minority
        test_indeces = test_indeces_majority + test_indeces_minority
        oracle_indeces = oracle_indeces_minority

        train_indeces.sort()
        val_indeces.sort()
        test_indeces.sort()
        oracle_indeces.sort()

        combined_df.loc[train_indeces, 'split'] = 'train'
        combined_df.loc[val_indeces, 'split'] = 'val'
        combined_df.loc[test_indeces, 'split'] = 'test'
        combined_df.loc[oracle_indeces, 'split'] = 'oracle'

        X_train = X[train_indeces]
        X_val = X[val_indeces]
        X_test = X[test_indeces]
        X_oracle = X[oracle_indeces]
        y_train = labels_and_static[train_indeces]
        y_val = labels_and_static[val_indeces]
        y_test = labels_and_static[test_indeces]
        y_oracle = labels_and_static[oracle_indeces]

        return {'train': (X_train, y_train), 'val': (X_val, y_val), 'test' : (X_test,y_test), 'oracle': (X_oracle,y_oracle),'static' : combined_df} 

    def stratified_split_dataset(self, X_in, y, c_in, intersectional, static_var, train_fraction, val_fraction, oracle_fraction, 
                                 oracle_min,intersectional_min_threshold,intersectional_max_threshold):
        #  train_fraction=0.3, val_fraction=0.1
        """
        Splits dataset into stratified train/val/test.

        Args:
            X_in: dict with where entries are asuumed to share the first dimension.
                Includes at least 'X' and 'feature_names' keys.

        Returns:
            X_dict: dictionary with X_train, X_val, X_test
            y_dict: dictionary with y_train, y_val, y_test
        """
        key_list = []
        input_list = []
        for key, value in X_in.items():
            if key != 'feature_names':
                key_list.append(key)
                input_list.append(value)

        X = np.stack(input_list, axis=1)
        # TODO: MOVE THIS TO UTIL FUNCTION

        # c=self.select_static_features(c_in,static_var)

        # y = np.concatenate((np.expand_dims(y,1), c['c']), axis=1)
        # y = np.concatenate((y.reshape(-1, 1), c.reshape(-1, 1)), axis=1)

        # X_train, X_val, X_test, X_oracle, y_train, y_val, y_test, y_oracle,patients_train, patients_val, patients_test, patients_oracle = self.majority_minority_split(
        #                                                                                             X, c_in, y,static_var, majority_labels=majority_labels, minority_labels=minority_labels,
        #                                                                                             oracle_fraction= oracle_fraction, train_fraction=train_fraction, val_fraction=val_fraction)
        if intersectional:
            results = self.intersectional_split(X, c_in, y, train_fraction, val_fraction, oracle_fraction,oracle_min,intersectional_min_threshold,intersectional_max_threshold)
            # outputs train, val, test, oracle, static
        else:
            results = self.majority_minority_split(X, c_in, y, static_var, train_fraction = train_fraction, val_fraction = val_fraction, increment=0.1)
        
        X_train, y_train = results['train']
        X_val, y_val = results['val']
    

        
        X_dict = {}
        y_dict = {
            'y_train': y_train[:,0],
            'y_val': y_val[:,0]
        }
        c_dict = {
            'c_train': y_train[:,1:],
            'c_val': y_val[:,1:],
            'feature_names': c_in['feature_names']
        }

        for idx, key in enumerate(key_list):
            X_dict[F'{key}_train'] = X_train[:,idx,...]
            X_dict[F'{key}_val'] = X_val[:, idx, ...]
            if  isinstance(results['test'], list):
                for test_idx, (X_test, y_test) in enumerate(results['test']):
                    X_dict[F'{key}_test_{test_idx}'] = X_test[:, idx, ...]
                    y_dict[F'y_test_{test_idx}']= y_test[:,0]
                    c_dict[F'c_test_{test_idx}']= y_test[:,1:]
            else:
                X_test, y_test = results['test']
                X_dict[F'{key}_test'] = X_test[:, idx, ...]
                y_dict['y_test'] = y_test[:,0]
                c_dict['c_test'] = y_test[:,1:]
        
            if  isinstance(results['oracle'], list):
                for oracle_idx, (X_oracle, y_oracle) in enumerate(results['oracle']):
                    X_dict[F'{key}_oracle_{oracle_idx}'] = X_oracle[:, idx, ...]
                    y_dict[F'y_oracle_{oracle_idx}'] = y_oracle[:,0]
                    c_dict[F'c_oracle_{oracle_idx}'] = y_oracle[:,1:]
            else:
                X_oracle, y_oracle = results['oracle']
                X_dict[F'{key}_oracle'] = X_oracle[:, idx, ...]
                y_dict['y_oracle'] = y_oracle[:,0]
                c_dict['c_oracle'] = y_oracle[:,1:]

        X_dict['feature_names'] = X_in['feature_names']
        y_dict.update(c_dict)


        if intersectional:
            return X_dict, y_dict , results['static']
        else:
            return X_dict, y_dict

    def preprocess_variables(self, x, preprocessors, split =  True):
        """
        Args:
            x: variables to preprocess. may be an array or dict, depending on
               preprocessors that are called
            preprocessors: ordered list of preprocessing methods

        Returns:
            x: preprocessed variables
        """
        # Preprocess with steps defined in preprocessors
        for process_step in preprocessors:
            x = process_step.transform(x, split)

        return x

    def save_intermed(self, X, y, c):

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_dir = os.path.join(self.intermed_output_path,timestamp)
        os.makedirs(new_dir, exist_ok=True)

        intermed_features_path =  os.path.join(new_dir,
                                        F'X_{timestamp}_intermed.npz')
        intermed_labels_path =  os.path.join(new_dir,
                                        F'y_{timestamp}_intermed.npy')     
        intermed_static_path =  os.path.join(new_dir,
                                        F'c_{timestamp}_intermed.npz') 
        
        np.savez(intermed_features_path,
                 X_original=X['X_original'],X_imputed=X['X_imputed'],
                 m=X['m'], delta_t=X['delta_t'],
                 feature_names=X['feature_names'])
        np.save(intermed_labels_path, y)
        np.savez(intermed_static_path, **c)
        print('Saved intermed data!')
        print('Intermed features path:', intermed_features_path)
        print('Intermed labels path:', intermed_labels_path)
        print('Intermed static path:', intermed_static_path)
        
    def save_processed(self, X, y,static = None):
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_dir = os.path.join(self.processed_output_path,timestamp)
        os.makedirs(new_dir, exist_ok=True)

        processed_features_path = os.path.join(new_dir,
                            F'X_{timestamp}_processed')
        
        processed_labels_path= os.path.join(new_dir,
                            F'y_{timestamp}_processed')
        processed_static_path = os.path.join(new_dir,
                            F'static_{timestamp}_processed.csv')

        np.savez(processed_features_path, **X)
        np.savez(processed_labels_path, **y)
        print('Saved processed data!')
        if static is not None:
            static.to_csv(processed_static_path)
            print('Processed static path:', processed_static_path)

        print('Processed features path:', processed_features_path)
        print('Processed labels path:', processed_labels_path)


        return new_dir,timestamp


    def get_data(self,mode, train_fraction, val_fraction, oracle_fraction,
                 oracle_min,intersectional_min_threshold,intersectional_max_threshold,
                 standardize =False,
                 stratify= True, intersectional =True, split = True,
                 save_intermed_data = True, save_processed_data = True,
                 processed_timestamp = None,intermed_timestamp = None,
                 demographics_to_stratify_on = ['age_group','ethnicity','gender']):
        """
            Retrieves the preprocessed input features and labels for training.
            - to split into train, val, test, oracle: set stratify = True and intersectional = True and split = True
            - to split into train, val, test: set stratify = False and intersectional = False and split = True. 
                This will split the data into stratified (based on outcome and demographics_to_stratify_on) train, val, test partitions
                based on train and val fractions.
            - to avoid splitting, set split = False
            
            Args:
                mode (str): The mode to load the data. Options are 'raw', 'intermediate', or 'processed'.
                train_fraction (float): Fraction of data for training set
                val_fraction (float): Fraction of data for validation set
                oracle_fraction (float): Target fraction for oracle set (minority groups only)
                oracle_min (int): Minimum size for oracle set (minority groups only)
                intersectional_min_threshold (int): Minimum size for a group to be included
                intersectional_max_threshold (int): Threshold between majority and minority groups
                stratify (bool, optional): Whether to perform stratified splitting of the dataset. Defaults to True.
                intersectional (bool, optional): Whether to perform intersectional splitting of the dataset. Defaults to True.
                split (bool, optional): Whether to split the dataset into train test, valid, oracle. Defaults to True.
                save_intermed_data (bool, optional): Whether to save intermediate data. Defaults to True.
                save_processed_data (bool, optional): Whether to save processed data. Defaults to True.
                processed_timestamp (str, optional): The timestamp of the processed data to load. Defaults to None.
                intermed_timestamp (str, optional): The timestamp of the intermediate data to load. Defaults to None.
            
            Returns:
                X_dict_tf (dict): A dictionary of preprocessed input features for training.
                y_dict (dict): A dictionary of labels for training.
        """
        
        if mode == 'processed' and (processed_timestamp is None):
            raise ValueError("Cannot load processed data when processed_timestamp is not specified")
        if mode == 'intermediate' and (intermed_timestamp is None):
            raise ValueError("Cannot load intermediate data when intermed_timestamp is not specified")           
        if mode == 'processed' and save_processed_data:
            raise ValueError("Cannot save processed data when mode is 'processed'")
        if mode == 'processed' and save_intermed_data:
            raise ValueError("Cannot save intermed data when mode is 'processed'")  
        if mode == 'intermediate' and save_intermed_data:
            raise ValueError("Cannot save intermed data when mode is 'intermediate'")  
        if not split and (stratify or intersectional):
            raise ValueError("Cannot set 'stratify' or 'intersectional' to True when 'split' is False")
        if not stratify and not intersectional and split and oracle_fraction > 0:
            raise ValueError("Cannot set 'oracle_fraction' to a value greater than 0 when 'stratify' and 'intersectional' are False")
        if stratify and intersectional and oracle_fraction == 0:
            raise ValueError("Cannot set 'oracle_fraction' to 0 when 'stratify' and 'intersectional' are True")
        if stratify and intersectional and split:
            warnings.warn("This will split the data into train, val, test, oracle sets.")
        if not stratify and not intersectional and split:
            warnings.warn("This will split the data into train, val, test sets stratified on outcome and demographics_to_stratify_on, according to the train_fraction and val_fraction")    
        if mode == 'processed':
            # Load processed data
            processed_dir = os.path.join(self.processed_output_path, processed_timestamp)
            static_files = [f for f in os.listdir(processed_dir) if f.startswith('static')]
            
            X_dict = np.load(os.path.join(processed_dir, f'X_{processed_timestamp}_processed.npz'), allow_pickle=True) if processed_timestamp else None
            y_dict = np.load(os.path.join(processed_dir, f'y_{processed_timestamp}_processed.npz'), allow_pickle=True) if processed_timestamp else None
            static = pd.read_csv(os.path.join(processed_dir, f'static_{processed_timestamp}_processed.csv')) if static_files else None

        elif mode == 'intermediate':
            # Load intermediate data, process it, then optionally save it as processed data
            
            # dict with at least 'X': [N_patients, features, time_len]
            # and 'feature_names': [features]
            intermed_dir = os.path.join(self.intermed_output_path, intermed_timestamp)
            patients_intermed = np.load(os.path.join(intermed_dir,f'X_{intermed_timestamp}_intermed.npz'))
            self.patient_intermed = patients_intermed
            # array: [N_patients, labels]
            outcomes_intermed = np.load(os.path.join(intermed_dir,f'y_{intermed_timestamp}_intermed.npy'))
            self.outcomes_intermed = outcomes_intermed
            static_intermed = np.load(os.path.join(intermed_dir,f'c_{intermed_timestamp}_intermed.npz'),allow_pickle=True)
            self.static_intermed = static_intermed
            # if self.N_patients is None:
            #     self.N_patients = len(patients_intermed['X'])
        elif mode == 'raw':
            # Import raw data, process it, and save it as processed data

            # array: [N_patients, features, time_len], array: [features]
            static_intermed = self.get_static_data()
            self.static_intermed = static_intermed
            patients_intermed, feature_names = self.get_timeseries()
            self.patient_intermed = patients_intermed
            # array: [N_patients, labels]
            outcomes_intermed = self.get_outcomes()
            self.outcomes_intermed = outcomes_intermed
            # Preprocessing that is independent of split of dataset
            # dict with at least 'X': [N_patients, features, time_len]

            if save_intermed_data:
                self.save_intermed(patients_intermed, outcomes_intermed,
                                        static_intermed)
                logging.info('Saved intermediate data!')

            patients_truncated = self.select_features(patients_intermed, self.features)
            y = self.select_task(outcomes_intermed, self.task_name)
            if split:
                if stratify and intersectional:
                    X_dict, y_dict, static = self.stratified_split_dataset(patients_truncated, y, static_intermed, intersectional,
                                                                            self.static_var, train_fraction, val_fraction, 
                                                                            oracle_fraction,
                                                                            oracle_min, intersectional_min_threshold, intersectional_max_threshold)
                elif stratify:  # TODO: need to make this work and check what it does.
                    X_dict, y_dict = self.stratified_split_dataset(patients_truncated, y, static_intermed, intersectional,
                                                                    self.static_var, train_fraction, val_fraction, 
                                                                    oracle_fraction,
                                                                    oracle_min, intersectional_min_threshold, intersectional_max_threshold)
                    static = None
                else:
                    test_fraction = 1 - train_fraction - val_fraction
                    X_dict, y_dict = self.split_dataset(patients_truncated, y, static_intermed, val_fraction, test_fraction,demographics_to_stratify_on)
                    static = None

            else:
                X_dict = patients_truncated
                y_dict = {'y': y.flatten(), 'c': static_intermed['c'], 'feature_names': static_intermed['feature_names']}
                static = None

            # Standardize the input features
            if standardize:
                # Currently. standardizatin is optional.
                # also, standardization is done on the raw data, not the imputed ones, and it return null values. 
                # The current implementation is not correct as it overrides all the imputation done in the previous steps.
                X_dict = self.preprocess_variables(X_dict, self.get_input_preprocessors(),split)

            if save_processed_data:
                new_dir, timestamp = self.save_processed(X_dict, y_dict,static)
                logging.info('Saved processed data!')
                # Save metadata
                metadata = {
                    'task_name': self.task_name,
                    'data_name': self.data_name,
                    'timestamp': timestamp,
                    'stratify' : stratify,
                    'intersectional': intersectional,
                    'split' : split,
                    'standardize': standardize,
                    'train_fraction': train_fraction,
                    'val_fraction': val_fraction,
                    'oracle_fraction': oracle_fraction,
                    'oracle_min': oracle_min,
                    'intersectional_min_threshold': intersectional_min_threshold,
                    'intersectional_max_threshold': intersectional_max_threshold,
                    'simple imputation': self.imputation,
                }
                metadata_path = os.path.join(new_dir, 'metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
        return X_dict, y_dict, static


