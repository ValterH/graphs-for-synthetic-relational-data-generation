import argparse

import pandas as pd
import numpy as np


from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sdmetrics.utils import HyperTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.utils import merge_children, load_tables, load_metadata, get_root_table, add_number_of_children

"""
How to call:

python src/eval/eval_classifier.py --dataset [NAME_OF_DATASET] --method [SYNTHETIC_DATA_GENERATION_METHOD]

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
                             max_items = 100000, **kwargs):

    seed = kwargs.get('seed', 42)
    transformed_original = original.copy()
    transformed_synthetic = synthetic.copy()

    column_names = transformed_original.columns.to_list()
    transformed_original = transformed_original.reindex(column_names, axis=1)
    transformed_synthetic = transformed_synthetic.reindex(column_names, axis=1)

    n = min(max_items, transformed_original.shape[0], transformed_synthetic.shape[0])
    transformed_original = transformed_original.sample(n=n, random_state=seed, replace=False)
    transformed_synthetic = transformed_synthetic.sample(n=n, random_state=seed, replace=False)

    # synthetic labels are 1 as this is what we are interested in (for precision and recall)
    y = np.hstack([
        np.zeros(transformed_original.shape[0]),
        np.ones(transformed_synthetic.shape[0])
    ])
    X = pd.concat([transformed_original, transformed_synthetic], axis=0)

    ht = CustomHyperTransformer()
    X = ht.fit_transform(X)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])


    def cross_validation(model, X, y, cv=5):
        np.random.seed(seed)
        folds = np.random.randint(0, cv, size=X.shape[0])
        scores = {
            'zero_one': [],
            'log_loss': [],
        }
        for i in range(cv):
            model.fit(X[folds != i], y[folds != i])
            probs = model.predict_proba(X[folds == i])
            preds = probs.argmax(axis=1)
            scores['zero_one'].append((preds == y[folds == i]).astype(int))
            scores['log_loss'].append(log_loss(y[folds == i], probs))
        scores['zero_one'] = np.hstack(scores['zero_one']).tolist()
        scores['log_loss'] = np.hstack(scores['log_loss'])
        return scores

    return cross_validation(model, X, y, cv=5)


def parent_child_discriminative_detection(original, synthetic, clf=LogisticRegression(solver='lbfgs', max_iter=100), 
                                          max_items = 100000, **kwargs):
    metadata = kwargs.get('metadata', None)
    root_table = kwargs.get('root_table', None)

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
    
    parser.add_argument('--method', type=str, default='ours/mlp_gnn', 
                        help='Specify the synthetic data generation method to evaluate')
    

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


def logistic_detection(dataset, method, seed=0):
    clf = LogisticRegression(solver='lbfgs', max_iter=250, random_state=seed)

    metadata = load_metadata(dataset)
    root_table = get_root_table(dataset)

    tables_synthetic = load_tables(dataset, data_type=f'synthetic/{method}')
    tables_original = load_tables(dataset, split='train')
    tables_synthetic_children = dict()
    tables_original_children = dict()
    for table in metadata.get_tables():
        # convert to correct data type
        for column, data_type in metadata.get_dtypes(table).items():
            if data_type == 'object':
                # convert both to the same datatype before coneverting to string
                original_dtype = tables_original[table][column].dtype
                tables_synthetic[table][column] = tables_synthetic[table][column].astype(original_dtype)
                tables_synthetic[table][column] = tables_synthetic[table][column].astype(str)
                tables_original[table][column] = tables_original[table][column].astype(str)
            elif data_type == 'datetime':
                tables_synthetic[table][column] = pd.to_numeric(pd.to_datetime(tables_synthetic[table][column]))
                tables_original[table][column] = pd.to_numeric(pd.to_datetime(tables_original[table][column]))
            else:
                tables_synthetic[table][column] = tables_synthetic[table][column].astype(data_type)
                tables_original[table][column] = tables_original[table][column].astype(data_type)
        if metadata.get_children(table):
            tables_synthetic_children[table] = add_number_of_children(table, metadata, tables_synthetic)
            tables_original_children[table] = add_number_of_children(table, metadata, tables_original)
        else:
            tables_synthetic_children[table] = tables_synthetic[table].copy()
            tables_original_children[table] = tables_original[table].copy()

    pc_results = parent_child_discriminative_detection(tables_original, tables_synthetic, 
                                                       clf=clf, metadata=metadata, root_table=root_table, seed=seed)
    
    pc_children_results = parent_child_discriminative_detection(tables_original_children, tables_synthetic_children, 
                                                                clf=clf, metadata=metadata, root_table=root_table, seed=seed)
    
    results = {
        'parent_child': pc_results['zero_one'],
        'parent_child_children': pc_children_results['zero_one']
        }
    for table in metadata.get_tables():
        original = drop_ids(tables_original[table].copy(), table, metadata)
        synthetic = drop_ids(tables_synthetic[table].copy(), table, metadata)
        results[table] = discriminative_detection(original, synthetic, clf=clf, seed=seed)['zero_one']
        if metadata.get_children(table):
            original_children = drop_ids(tables_original_children[table].copy(), table, metadata)
            synthetic_children = drop_ids(tables_synthetic_children[table].copy(), table, metadata)
            results[f'{table}_children'] = discriminative_detection(original_children, synthetic_children, clf=clf, seed=seed)['zero_one']
    
    return results


def main():
    args = get_args()   
    results = logistic_detection(args.dataset, args.method)
    for key, value in results.items():
        print(f'{key :<22}: {np.mean(value):.3} ± {np.std(value) / np.sqrt(len(value)):.3}')

if __name__ == "__main__":
    main()