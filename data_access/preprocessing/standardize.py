"""
2021 Simon Bing, ETHZ, MPI IS
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_access.preprocessing.base_processor import BaseProcessor

class Standardize(BaseProcessor):
    """
    Standardization is done on the raw data, not the imputed ones,
    and it the standardized raw data along with null values.  
    The current implementation is not correct as it overrides all the
    imputation done in the previous steps.
    
    
    """
    def __init__(self):
        super().__init__()


    # def transform(self, x):
    #     """
    #     Standardize input features.

    #     Args:
    #         x: dict with X_train, X_val, X_test arrays of dim.: [N, features, time_len]

    #     Returns:
    #         X_dict_tf: dict of transformed inputs.
    #     """
    #     # Check if missing values mask exists and replace zeros with nans
        
    #     if 'm_train' in x.keys():
    #         X_train = np.where(x['m_train'], x['X_train'], np.nan)
    #         X_val = np.where(x['m_val'], x['X_val'], np.nan)
    #         X_test = np.where(x['m_test'], x['X_test'], np.nan)
    #         X_oracle = np.where(x['m_oracle'], x['X_oracle'], np.nan)
    #     else:
    #         X_train = x['X_train']
    #         X_val = x['X_val']
    #         X_test = x['X_test']
    #         X_oracle = x['X_oracle']


    #     n_features = X_train.shape[1]
    #     time_len = X_train.shape[2]

    #     # Flatten time dimension
    #     X_train_reshape = np.reshape(X_train.transpose(0,2,1), (-1, n_features))
    #     X_val_reshape = np.reshape(X_val.transpose(0, 2, 1), (-1, n_features))
    #     X_test_reshape = np.reshape(X_test.transpose(0, 2, 1), (-1, n_features))
    #     X_oracle_reshape = np.reshape(X_oracle.transpose(0, 2, 1), (-1, n_features))

    #     # Fit scaler
    #     scaler = StandardScaler()
    #     scaler.fit(X_train_reshape)

    #     # Transform data
    #     X_train_reshape_tf = scaler.transform(X_train_reshape)
    #     X_val_reshape_tf = scaler.transform(X_val_reshape)
    #     X_test_reshape_tf = scaler.transform(X_test_reshape)
    #     X_oracle_reshape_tf = scaler.transform(X_oracle_reshape)
        
    #     # Recover original dimensions
    #     X_train_tf = np.reshape(X_train_reshape_tf, X_train.transpose(0,2,1).shape).transpose(0,2,1)
    #     X_val_tf = np.reshape(X_val_reshape_tf, X_val.transpose(0, 2, 1).shape).transpose(0,2,1)
    #     X_test_tf = np.reshape(X_test_reshape_tf, X_test.transpose(0, 2, 1).shape).transpose(0,2,1)
    #     X_oracle_tf = np.reshape(X_oracle_reshape_tf, X_oracle.transpose(0, 2, 1).shape).transpose(0,2,1)
        
    #     X_dict_tf = x.copy()
    #     X_dict_tf['X_train'] = np.nan_to_num(X_train_tf)
    #     X_dict_tf['X_val'] = np.nan_to_num(X_val_tf)
    #     X_dict_tf['X_test'] = np.nan_to_num(X_test_tf)
    #     X_dict_tf['X_oracle'] = np.nan_to_num(X_oracle_tf)

    #     return X_dict_tf
    

    # def transform(self, x):

    #     def apply_mask_if_exists(data_key, mask_key):
    #         return np.where(x[mask_key], x[data_key], np.nan) if mask_key in x else x[data_key]

    #     # Initialize the transformed dictionary
    #     X_dict_tf = x.copy()

    #     # Process train and validation sets
    #     X_train = apply_mask_if_exists('X_train', 'm_train')
    #     X_val = apply_mask_if_exists('X_val', 'm_val')


    #     # Process dynamic test and oracle sets
    #     test_keys = [key for key in x.keys() if key.startswith('X_test_')]
    #     oracle_keys = [key for key in x.keys() if key.startswith('X_oracle_')]



    #     if len(test_keys) > 0:
    #         X_tests = []
    #         for key in test_keys:
    #             mask_key = key.replace('X_test_', 'm_test_')
    #             X_tests.append(apply_mask_if_exists(key, mask_key))

    #         X_oracles = []
    #         for key in oracle_keys:
    #             mask_key = key.replace('X_oracle_', 'm_oracle_')
    #             X_oracles.append(apply_mask_if_exists(key, mask_key))
    #     else:
    #         X_test = apply_mask_if_exists('X_test', 'm_test')
    #         X_oracle = apply_mask_if_exists('X_oracle', 'm_oracle')


    #     # Determine number of features and time length
    #     n_features = X_train.shape[1]
    #     time_len = X_train.shape[2]

    #     # Flatten time dimension for scaling
    #     def reshape_for_scaling(X):
    #         return np.reshape(X.transpose(0, 2, 1), (-1, n_features))

    #     X_train_reshape = reshape_for_scaling(X_train)
    #     X_val_reshape = reshape_for_scaling(X_val)
    #     if len(test_keys) > 0:
    #         X_tests_reshape = [reshape_for_scaling(X_test) for X_test in X_tests]
    #         X_oracles_reshape = [reshape_for_scaling(X_oracle) for X_oracle in X_oracles]
    #     else:
    #         X_test_reshape = reshape_for_scaling(X_test)
    #         X_oracle_reshape = reshape_for_scaling(X_oracle)
    
    #     # Fit scaler
    #     scaler = StandardScaler()
    #     scaler.fit(X_train_reshape)

    #     # Transform data
    #     X_train_reshape_tf = scaler.transform(X_train_reshape)
    #     X_val_reshape_tf = scaler.transform(X_val_reshape)
    #     if len(test_keys) > 0:
    #         X_tests_reshape_tf = [scaler.transform(X_test_reshape) for X_test_reshape in X_tests_reshape]
    #         X_oracles_reshape_tf = [scaler.transform(X_oracle_reshape) for X_oracle_reshape in X_oracles_reshape]
    #     else:
    #         X_test_reshape_tf = scaler.transform(X_test_reshape)
    #         X_oracle_reshape_tf = scaler.transform(X_oracle_reshape)


    #     # Recover original dimensions
    #     def recover_dimensions(X_reshape_tf, original_shape):
    #         return np.reshape(X_reshape_tf, original_shape).transpose(0, 2, 1)

    #     X_train_tf = recover_dimensions(X_train_reshape_tf, X_train.transpose(0, 2, 1).shape)
    #     X_val_tf = recover_dimensions(X_val_reshape_tf, X_val.transpose(0, 2, 1).shape)
    #     if len(test_keys) > 0:
    #         X_tests_tf = [recover_dimensions(X_test_reshape_tf, X_test.transpose(0, 2, 1).shape) for X_test_reshape_tf, X_test in zip(X_tests_reshape_tf, X_tests)]
    #         X_oracles_tf = [recover_dimensions(X_oracle_reshape_tf, X_oracle.transpose(0, 2, 1).shape) for X_oracle_reshape_tf, X_oracle in zip(X_oracles_reshape_tf, X_oracles)]
    #     else:
    #         X_test_tf = recover_dimensions(X_test_reshape_tf, X_test.transpose(0, 2, 1).shape)
    #         X_oracle_tf = recover_dimensions(X_oracle_reshape_tf, X_oracle.transpose(0, 2, 1).shape)

    #     # Update the transformed dictionary
    #     X_dict_tf['X_train'] = np.nan_to_num(X_train_tf)
    #     X_dict_tf['X_val'] = np.nan_to_num(X_val_tf)
    #     if len(test_keys) > 0:
    #         for i, key in enumerate(test_keys):
    #             X_dict_tf[key] = np.nan_to_num(X_tests_tf[i])

    #         for i, key in enumerate(oracle_keys):
    #             X_dict_tf[key] = np.nan_to_num(X_oracles_tf[i])
    #     else:
    #         X_dict_tf['X_test'] = np.nan_to_num(X_test_tf)
    #         X_dict_tf['X_oracle'] = np.nan_to_num(X_oracle_tf)
            
    #     return X_dict_tf
    
    def transform(self, x,split = True):

        def apply_mask_if_exists(data_key, mask_key):
            return np.where(x[mask_key], x[data_key], np.nan) if mask_key in x else x[data_key]

        # Initialize the transformed dictionary
        X_dict_tf = x.copy()

        if split:

            # Process train and validation sets
            X_train = apply_mask_if_exists('X_imputed_train', 'm_train')
            X_val = apply_mask_if_exists('X_imputed_val', 'm_val')
            X_test = apply_mask_if_exists('X_imputed_test', 'm_test')
            X_oracle = apply_mask_if_exists('X_imputed_oracle', 'm_oracle')


            # Determine number of features and time length
            n_features = X_train.shape[1]
            time_len = X_train.shape[2]

            # Flatten time dimension for scaling
            def reshape_for_scaling(X):
                return np.reshape(X.transpose(0, 2, 1), (-1, n_features))

            X_train_reshape = reshape_for_scaling(X_train)
            X_val_reshape = reshape_for_scaling(X_val)
            X_test_reshape = reshape_for_scaling(X_test)
            X_oracle_reshape = reshape_for_scaling(X_oracle)

            # Fit scaler
            scaler = StandardScaler()
            scaler.fit(X_train_reshape)

            # Transform data
            X_train_reshape_tf = scaler.transform(X_train_reshape)
            X_val_reshape_tf = scaler.transform(X_val_reshape)
            X_test_reshape_tf = scaler.transform(X_test_reshape)
            X_oracle_reshape_tf = scaler.transform(X_oracle_reshape)


            # Recover original dimensions
            def recover_dimensions(X_reshape_tf, original_shape):
                return np.reshape(X_reshape_tf, original_shape).transpose(0, 2, 1)

            X_train_tf = recover_dimensions(X_train_reshape_tf, X_train.transpose(0, 2, 1).shape)
            X_val_tf = recover_dimensions(X_val_reshape_tf, X_val.transpose(0, 2, 1).shape)
            X_test_tf = recover_dimensions(X_test_reshape_tf, X_test.transpose(0, 2, 1).shape)
            X_oracle_tf = recover_dimensions(X_oracle_reshape_tf, X_oracle.transpose(0, 2, 1).shape)

            # Update the transformed dictionary
            X_dict_tf['X_imputed_train'] = np.nan_to_num(X_train_tf)
            X_dict_tf['X_imputed_val'] = np.nan_to_num(X_val_tf)
            X_dict_tf['X_imputed_test'] = np.nan_to_num(X_test_tf)
            X_dict_tf['X_imputed_oracle'] = np.nan_to_num(X_oracle_tf)
            
        else:
            # Process the entire dataset without splits
            X = x['X_imputed']
            # Determine number of features and time length
            n_features = X.shape[1]
            time_len = X.shape[2]

            # Flatten time dimension for scaling
            X_reshape = np.reshape(X.transpose(0, 2, 1), (-1, n_features))

            # Fit scaler
            scaler = StandardScaler()
            scaler.fit(X_reshape)

            # Transform data
            X_reshape_tf = scaler.transform(X_reshape)

            # Recover original dimensions
            X_tf = np.reshape(X_reshape_tf, X.transpose(0, 2, 1).shape).transpose(0, 2, 1)

            # Update the transformed dictionary
            # X_dict_tf['X'] = np.nan_to_num(X_tf)
            X_dict_tf['X_imputed_scaled'] = X_tf
        return X_dict_tf
    


    # def transform(self, x, split=True):
    #     def apply_mask_if_exists(data_key, mask_key):
    #         return np.where(x[mask_key], x[data_key], np.nan) if mask_key in x else x[data_key]

    #     # Initialize the transformed dictionary
    #     X_dict_tf = x.copy()

    #     if split:
    #         # Process train and validation sets
    #         X_train = apply_mask_if_exists('X_train', 'm_train')
    #         X_val = apply_mask_if_exists('X_val', 'm_val')

    #         # Process dynamic test and oracle sets
    #         test_keys = [key for key in x.keys() if key.startswith('X_test_')]
    #         oracle_keys = [key for key in x.keys() if key.startswith('X_oracle_')]

    #         if len(test_keys) > 0:
    #             X_tests = []
    #             for key in test_keys:
    #                 mask_key = key.replace('X_test_', 'm_test_')
    #                 X_tests.append(apply_mask_if_exists(key, mask_key))

    #             X_oracles = []
    #             for key in oracle_keys:
    #                 mask_key = key.replace('X_oracle_', 'm_oracle_')
    #                 X_oracles.append(apply_mask_if_exists(key, mask_key))
    #         else:
    #             X_test = apply_mask_if_exists('X_test', 'm_test')
    #             X_oracle = apply_mask_if_exists('X_oracle', 'm_oracle')

    #         # Determine number of features and time length
    #         n_features = X_train.shape[1]
    #         time_len = X_train.shape[2]

    #         # Flatten time dimension for scaling
    #         def reshape_for_scaling(X):
    #             return np.reshape(X.transpose(0, 2, 1), (-1, n_features))

    #         X_train_reshape = reshape_for_scaling(X_train)
    #         X_val_reshape = reshape_for_scaling(X_val)
    #         if len(test_keys) > 0:
    #             X_tests_reshape = [reshape_for_scaling(X_test) for X_test in X_tests]
    #             X_oracles_reshape = [reshape_for_scaling(X_oracle) for X_oracle in X_oracles]
    #         else:
    #             X_test_reshape = reshape_for_scaling(X_test)
    #             X_oracle_reshape = reshape_for_scaling(X_oracle)

    #         # Fit scaler
    #         scaler = StandardScaler()
    #         scaler.fit(X_train_reshape)

    #         # Transform data
    #         X_train_reshape_tf = scaler.transform(X_train_reshape)
    #         X_val_reshape_tf = scaler.transform(X_val_reshape)
    #         if len(test_keys) > 0:
    #             X_tests_reshape_tf = [scaler.transform(X_test_reshape) for X_test_reshape in X_tests_reshape]
    #             X_oracles_reshape_tf = [scaler.transform(X_oracle_reshape) for X_oracle_reshape in X_oracles_reshape]
    #         else:
    #             X_test_reshape_tf = scaler.transform(X_test_reshape)
    #             X_oracle_reshape_tf = scaler.transform(X_oracle_reshape)

    #         # Recover original dimensions
    #         def recover_dimensions(X_reshape_tf, original_shape):
    #             return np.reshape(X_reshape_tf, original_shape).transpose(0, 2, 1)

    #         X_train_tf = recover_dimensions(X_train_reshape_tf, X_train.transpose(0, 2, 1).shape)
    #         X_val_tf = recover_dimensions(X_val_reshape_tf, X_val.transpose(0, 2, 1).shape)
    #         if len(test_keys) > 0:
    #             X_tests_tf = [recover_dimensions(X_test_reshape_tf, X_test.transpose(0, 2, 1).shape) for X_test_reshape_tf, X_test in zip(X_tests_reshape_tf, X_tests)]
    #             X_oracles_tf = [recover_dimensions(X_oracle_reshape_tf, X_oracle.transpose(0, 2, 1).shape) for X_oracle_reshape_tf, X_oracle in zip(X_oracles_reshape_tf, X_oracles)]
    #         else:
    #             X_test_tf = recover_dimensions(X_test_reshape_tf, X_test.transpose(0, 2, 1).shape)
    #             X_oracle_tf = recover_dimensions(X_oracle_reshape_tf, X_oracle.transpose(0, 2, 1).shape)

    #         # Update the transformed dictionary
    #         X_dict_tf['X_train'] = np.nan_to_num(X_train_tf)
    #         X_dict_tf['X_val'] = np.nan_to_num(X_val_tf)
    #         if len(test_keys) > 0:
    #             for i, key in enumerate(test_keys):
    #                 X_dict_tf[key] = np.nan_to_num(X_tests_tf[i])

    #             for i, key in enumerate(oracle_keys):
    #                 X_dict_tf[key] = np.nan_to_num(X_oracles_tf[i])
    #         else:
    #             X_dict_tf['X_test'] = np.nan_to_num(X_test_tf)
    #             X_dict_tf['X_oracle'] = np.nan_to_num(X_oracle_tf)
    #     else:
    #         # Process the entire dataset without splits
    #         # X = apply_mask_if_exists('X_imputed', 'm')
    #         X = x['X_imputed']
    #         # Determine number of features and time length
    #         n_features = X.shape[1]
    #         time_len = X.shape[2]

    #         # Flatten time dimension for scaling
    #         X_reshape = np.reshape(X.transpose(0, 2, 1), (-1, n_features))

    #         # Fit scaler
    #         scaler = StandardScaler()
    #         scaler.fit(X_reshape)

    #         # Transform data
    #         X_reshape_tf = scaler.transform(X_reshape)

    #         # Recover original dimensions
    #         X_tf = np.reshape(X_reshape_tf, X.transpose(0, 2, 1).shape).transpose(0, 2, 1)

    #         # Update the transformed dictionary
    #         # X_dict_tf['X'] = np.nan_to_num(X_tf)
    #         X_dict_tf['X_imputed_scaled'] = X_tf
    #     return X_dict_tf