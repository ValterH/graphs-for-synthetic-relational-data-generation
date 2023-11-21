import os
import pandas as pd
import numpy as np
import pickle
import argparse

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from sdmetrics.utils import HyperTransformer

from src.data.utils import merge_children, load_tables, sdv_metadata

"""
How to call:

python src/eval/eval_classifier.py --model [MODEL_FOR_CLASSIFICATION] --dataset [NAME_OF_DATASET] --method [SYNTHETIC_DATA_GENERATION_METHOD]

"""



class CustomHyperTransformer(HyperTransformer):
    def fit(self, data):
        """Fit the HyperTransformer to the given data.

        Args:
            data (pandas.DataFrame):
                The data to transform.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            kind = data[field].dropna().infer_objects().dtype.kind
            self.column_kind[field] = kind

            if kind == 'i' or kind == 'f':
                # Numerical column.
                self.column_transforms[field] = {'mean': data[field].mean()}
            elif kind == 'b':
                # Boolean column.
                numeric = pd.to_numeric(data[field], errors='coerce').astype(float)
                self.column_transforms[field] = {'mode': numeric.mode().iloc[0]}
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                enc = OneHotEncoder(handle_unknown='ignore')
                enc.fit(col_data)
                self.column_transforms[field] = {'one_hot_encoder': enc}
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(
                    data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                self.column_transforms[field] = {'mean': pd.Series(integers).mean()}

    def transform(self, data):
        """Transform the given data based on the data type of each column.

        Args:
            data (pandas.DataFrame):
                The data to transform.

        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            transform_info = self.column_transforms[field]

            kind = self.column_kind[field]
            if kind == 'i' or kind == 'f':
                # Numerical column.
                data[field] = data[field].fillna(transform_info['mean'])
            elif kind == 'b':
                # Boolean column.
                data[field] = pd.to_numeric(data[field], errors='coerce').astype(float)
                data[field] = data[field].fillna(transform_info['mode'])
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field].astype("object")})
                out = transform_info['one_hot_encoder'].transform(col_data).toarray()
                transformed = pd.DataFrame(
                    out, columns=[f'{field}_{i}' for i in range(np.shape(out)[1])])
                data = data.drop(columns=[field])
                data = pd.concat([data, transformed.set_index(data.index)], axis=1)
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(
                    data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                data[field] = pd.Series(integers)
                data[field] = data[field].fillna(transform_info['mean'])

        return data
    

def drop_ids(table, table_name, metadata):
    for field, values in metadata.to_dict()['tables'][table_name]['fields'].items():
        if 'ref' in values:
            foreign_key = field
            for column in table.columns:
                if foreign_key in column:
                    table.drop(column, axis=1, inplace=True)
    pk = metadata.get_primary_key(table_name)
    for column in table.columns:
        if pk in column:
            table.drop(column, axis=1, inplace=True)
    return table

    
def discriminative_detection(original_test, synthetic_test, original_train, 
                             synthetic_train, clf=LogisticRegression(solver='lbfgs', max_iter=100), 
                             max_items = 100000, save_path = None, **kwargs):

    transformed_original_train = original_train.copy()
    transformed_synthetic_train = synthetic_train.copy()
    transformed_original_test = original_test.copy()
    transformed_synthetic_test = synthetic_test.copy()

    column_names = transformed_original_train.columns.to_list()
    transformed_original_test = transformed_original_test.reindex(column_names, axis=1)
    transformed_synthetic_train = transformed_synthetic_train.reindex(column_names, axis=1)
    transformed_synthetic_test = transformed_synthetic_test.reindex(column_names, axis=1)

    if 'Date' in column_names:
        transformed_original_train.drop('Date', axis=1, inplace=True)
        transformed_synthetic_train.drop('Date', axis=1, inplace=True)
        transformed_original_test.drop('Date', axis=1, inplace=True)
        transformed_synthetic_test.drop('Date', axis=1, inplace=True)

    # resample original test and synthetic test to same size
    n = min(len(transformed_original_test), len(transformed_synthetic_test))
    mask_original = np.zeros(len(transformed_original_test), dtype=bool)
    mask_original[:n] = True
    mask_original = np.random.permutation(mask_original)
    mask_synthetic = np.zeros(len(transformed_synthetic_test), dtype=bool)
    mask_synthetic[:n] = True
    mask_synthetic = np.random.permutation(mask_synthetic)

    # apply the mask
    transformed_original_test = transformed_original_test[mask_original]
    transformed_synthetic_test = transformed_synthetic_test[mask_synthetic]

    ht = CustomHyperTransformer()
    transformed_original_train = ht.fit_transform(transformed_original_train)

    transformed_original_train = transformed_original_train.to_numpy()
    transformed_original_test = ht.transform(transformed_original_test).to_numpy()
    transformed_synthetic_train = ht.transform(transformed_synthetic_train).to_numpy()
    transformed_synthetic_test = ht.transform(transformed_synthetic_test).to_numpy()

    X_train = np.concatenate([transformed_original_train, transformed_synthetic_train])
    X_test = np.concatenate([transformed_original_test, transformed_synthetic_test])

    # synthetic labels are 1 as this is what we are interested in (for precision and recall)
    y_train = np.hstack([
        np.zeros(transformed_original_train.shape[0]),
        np.ones(transformed_synthetic_train.shape[0])
    ])
    y_test = np.hstack([
        np.zeros(transformed_original_test.shape[0]),
        np.ones(transformed_synthetic_test.shape[0])
    ])

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)

    results = {
        'zero_one': (y_test == y_pred).astype(int).tolist(),
        'log_loss': log_loss(y_test, probs),
        'accuracy': accuracy_score(y_test, y_pred)
    }

    return results


def parent_child_discriminative_detection(original_test, synthetic_test, original_train, 
                                          synthetic_train, clf=LogisticRegression(solver='lbfgs', max_iter=100), 
                                          max_items = 100000, **kwargs):
    metadata = kwargs.get('metadata', None)
    root_table = kwargs.get('root_table', None)
    print(root_table)
    print(metadata)

    # join parent and child tables based on the metadata
    original_train = merge_children(original_train, metadata, root_table)
    synthetic_train = merge_children(synthetic_train, metadata, root_table)
    original_test = merge_children(original_test, metadata, root_table)
    synthetic_test = merge_children(synthetic_test, metadata, root_table)

    # drop all foreign and primary keys
    for table in metadata.to_dict()['tables'].keys():
        drop_ids(original_train, table, metadata)
        drop_ids(synthetic_train, table, metadata)
        drop_ids(original_test, table, metadata)
        drop_ids(synthetic_test, table, metadata)    
    
    return discriminative_detection(original_test, synthetic_test, original_train, 
                                    synthetic_train, clf=clf, max_items=max_items, **kwargs)


def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Which model to use for detection.')

    # dataset to evaluate
    parser.add_argument('--dataset', type=str, default='rossmann-store-sales',
                        help='Specify the dataset to evaluate.')
    
    parser.add_argument('--method', type=str, default='sdv', 
                        help='Specify the synthetic data generation method to evaluate')
    
    parser.add_argument('--model', type=str, choices=['logistic', 'xgboost'], default='xgboost',
                        help='Specify the classification model')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()    
    if args.model == 'logistic':
        clf = LogisticRegression(solver='lbfgs', max_iter=100)
    elif args.model == 'xgboost':
        clf = XGBClassifier(random_state=42)
    else:
        raise ValueError('Model not supported.')
    
    tables_train_synthetic = load_tables(args.dataset, 'train', data_type=f'synthetic/{args.method}')
    tables_test_synthetic = load_tables(args.dataset, 'test', data_type=f'synthetic/{args.method}')
    tables_train_original = load_tables(args.dataset, 'train')
    tables_test_original = load_tables(args.dataset, 'test')

    metadata = sdv_metadata.generate_metadata(args.dataset, tables_train_original)
    root_table=sdv_metadata.get_root_table(args.dataset)

    pc_results = parent_child_discriminative_detection(tables_test_original, tables_test_synthetic, 
                                                       tables_train_original, tables_train_synthetic, 
                                                       clf=clf, metadata=metadata, root_table=root_table)
    
    results = {'parent_child': pc_results['zero_one']}
    for table in tables_train_original.keys():
        original_train = drop_ids(tables_train_original[table].copy(), table, metadata)
        original_test = drop_ids(tables_test_original[table].copy(), table, metadata)
        synthetic_train = drop_ids(tables_train_synthetic[table].copy(), table, metadata)
        synthetic_test = drop_ids(tables_test_synthetic[table].copy(), table, metadata)
        results[table] = discriminative_detection(original_test, synthetic_test, original_train, 
                                                  synthetic_train, clf=clf)['zero_one']
    
    for key, value in results.items():
        print(f'{key :<12}: {np.mean(value):.3} ± {np.std(value) / np.sqrt(len(value)):.3}')