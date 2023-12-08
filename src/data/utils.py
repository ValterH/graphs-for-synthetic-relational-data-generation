import os

import pandas as pd
from sdv import Metadata

ROOT_DIR = 'graphs-for-synthetic-relational-data-generation'

def load_metadata(dataset_name):
    # get project root directory
    root_dir = os.getcwd().split(ROOT_DIR)[0] + ROOT_DIR
    path = os.path.join(root_dir, 'data', 'metadata', f'{dataset_name}_metadata.json')
    return Metadata(path)


def get_root_table(dataset_name):
    if dataset_name == "biodegradability":
        return "molecule"
    if dataset_name == "rossmann-store-sales":
        return "store"
    if dataset_name == "mutagenesis":
        return "molecule"
    if dataset_name == "coupon-purchase-prediction":
        return "user_list"
    if dataset_name == "telstra-competition-dataset":
        return "train"
    raise ValueError(f"Dataset {dataset_name} not supported")


def conditionally_sample(tables, metadata, root):
    parent = tables[root]
    children = metadata.get_children(root)
    for child in children:
        child_table = tables[child]
        fks = metadata.get_foreign_keys(root, child)
        for fk in fks:
            parent_fk = find_fk(root, fk, metadata)
            if parent_fk is None:
                continue
            parent_ids = parent[parent_fk].unique()
            tables[child] = child_table[child_table[fk].isin(parent_ids)]
            tables = conditionally_sample(tables, metadata, child)
    return tables


def read_original_tables(dataset_name):
    tables = {}
    for file_name in filter(lambda x: '.csv' in x, os.listdir(f'data/{dataset_name}')):
        table_name = file_name.split('.')[0]
        table = pd.read_csv(f'data/{dataset_name}/{file_name}')
        tables[table_name] = table
    return tables


def split_table(table, seed=42):
    table_train = table.copy()
    table_test = table.sample(frac=0.2, random_state=seed)
    return table_train, table_test


def save_tables(tables, dataset_name, split=None, data_type='original'):
    for table_name, table in tables.items():
        if split is None:
            table_path = f'data/{data_type}/{dataset_name}/{table_name}.csv'
        else:
            table_path = f'data/{data_type}/{dataset_name}/{table_name}_{split}.csv'
        if not os.path.exists(os.path.dirname(table_path)):
            os.makedirs(os.path.dirname(table_path))
        table.to_csv(table_path, index=False)


def load_tables(dataset_name, data_type='original', split=None):
    tables = {}
    data_path = f'data/{data_type}/{dataset_name}'
    for file_name in os.listdir(data_path):
        if not file_name.endswith('.csv'):
            continue
        if split is None:
            table_name = file_name.split('.')[0]
        else:
            if split not in file_name:
                continue
            table_name = file_name.split(f'_{split}')[0]
        table = pd.read_csv(f'{data_path}/{file_name}')
        tables[table_name] = table
    return tables


def prepare_dataset(dataset_name, seed=42):
    tables = read_original_tables(dataset_name)
    metadata = load_metadata(dataset_name)
    root_table = get_root_table(dataset_name)

    tables_train = {table_name: table.copy() for table_name, table in tables.items()}
    tables_test = {table_name: table.copy() for table_name, table in tables.items()}
    train_root, test_root = split_table(tables[root_table], seed=seed)
    tables_train[root_table] = train_root
    tables_test[root_table] = test_root
    tables_train = conditionally_sample(tables_train, metadata, root_table)
    tables_test = conditionally_sample(tables_test, metadata, root_table)
    save_tables(tables_train, dataset_name, split='train')
    save_tables(tables_test, dataset_name, split='test')


def find_fk(parent, reference, metadata):
    for field in metadata.to_dict()["tables"][parent]["fields"]:
        if field in reference:
            return field
    return None


def merge_children(tables, metadata, root, how="left"):
    parent = tables[root]
    children = metadata.get_children(root)
    for child in children:
        fks = metadata.get_foreign_keys(root, child)
        for i, fk in enumerate(fks):
            parent_fk = find_fk(root, fk, metadata)
            if parent_fk is None:
                continue
            child_table = merge_children(tables, metadata, child)
            if fk in parent.columns:
                parent = parent.merge(
                    child_table.drop_duplicates(),
                    left_on=fk,
                    right_on=fk,
                    how=how,
                    suffixes=("", f"_{child}_{i}"),
                )
            else:
                # this happens when there are 2 foreign keys from the same table
                # e.g. bond in biodegradabaility with fks atom_id and atom_id_2
                parent = parent.merge(
                    child_table.drop_duplicates(),
                    left_on=parent_fk,
                    right_on=fk,
                    how=how,
                    suffixes=("", f"_{child}_{i}"),
                )
    return parent