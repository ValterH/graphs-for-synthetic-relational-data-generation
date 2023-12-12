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
from sklearn.model_selection import train_test_split, cross_val_score

from src.data.utils import merge_children, load_tables, load_metadata, get_root_table

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

    
def discriminative_detection(original, synthetic, clf=LogisticRegression(solver='lbfgs', max_iter=100), 
                             max_items = 100000, save_path = None, **kwargs):

    transformed_original = original.copy()
    transformed_synthetic = synthetic.copy()

    column_names = transformed_original.columns.to_list()
    transformed_original = transformed_original.reindex(column_names, axis=1)
    transformed_synthetic = transformed_synthetic.reindex(column_names, axis=1)

    
    if 'Date' in column_names:
        transformed_original['Date'] = pd.to_numeric(pd.to_datetime(transformed_original['Date']))
        transformed_synthetic['Date'] = pd.to_numeric(pd.to_datetime(transformed_synthetic['Date']))
        # TODO: check if Date column is still problematic
        # transformed_original.drop('Date', axis=1, inplace=True)
        # transformed_synthetic.drop('Date', axis=1, inplace=True)

    # synthetic labels are 1 as this is what we are interested in (for precision and recall)
    y = np.hstack([
        np.zeros(transformed_original.shape[0]),
        np.ones(transformed_synthetic.shape[0])
    ])
    X = pd.concat([transformed_original, transformed_synthetic], axis=0)

    # TODO: we can do cross validation here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)


    ht = CustomHyperTransformer()
    X_train = ht.fit_transform(X_train)
    X_test = ht.transform(X_test)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)

    feature_importances = list(zip(X_train.columns, model['clf'].feature_importances_))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    for feature, importance in feature_importances:
        print(f'{feature :<12}: {importance:.3}')

    results = {
        'zero_one': (y_test == y_pred).astype(int).tolist(),
        'log_loss': log_loss(y_test, probs),
        'accuracy': accuracy_score(y_test, y_pred)
    }

    return results


def parent_child_discriminative_detection(original, synthetic, clf=LogisticRegression(solver='lbfgs', max_iter=100), 
                                          max_items = 100000, **kwargs):
    metadata = kwargs.get('metadata', None)
    root_table = kwargs.get('root_table', None)
    print(root_table)
    print(metadata)

    # join parent and child tables based on the metadata
    original = merge_children(original, metadata, root_table)
    synthetic = merge_children(synthetic, metadata, root_table)

    # drop all foreign and primary keys
    for table in metadata.to_dict()['tables'].keys():
        drop_ids(original, table, metadata)
        drop_ids(synthetic, table, metadata)
    
    return discriminative_detection(original, synthetic, clf=clf, max_items=max_items, **kwargs)


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
    
    tables_synthetic = load_tables(args.dataset, data_type=f'synthetic/{args.method}')
    tables_original = load_tables(args.dataset, split='train')

    metadata = load_metadata(args.dataset)
    root_table = get_root_table(args.dataset)

    pc_results = parent_child_discriminative_detection(tables_original, tables_synthetic, 
                                                       clf=clf, metadata=metadata, root_table=root_table)
    
    results = {'parent_child': pc_results['zero_one']}
    for table in tables_original.keys():
        original = drop_ids(tables_original[table].copy(), table, metadata)
        synthetic = drop_ids(tables_synthetic[table].copy(), table, metadata)
        results[table] = discriminative_detection(original, synthetic, clf=clf)['zero_one']
    
    for key, value in results.items():
        print(f'{key :<12}: {np.mean(value):.3} ± {np.std(value) / np.sqrt(len(value)):.3}')