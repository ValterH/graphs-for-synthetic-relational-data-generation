import os

import pandas as pd
from rike.utils import conditionally_sample
from rike.generation import sdv_metadata

def read_original_tables(dataset_name):
    tables = {}
    for file_name in os.listdir(f'data/{dataset_name}'):
        table_name = file_name.split('.')[0]
        table = pd.read_csv(f'data/{dataset_name}/{file_name}')
        tables[table_name] = table
    return tables


def split_table(table, seed=42):
    table_train = table.sample(frac=0.8, random_state=seed)
    table_test = table.drop(table_train.index)
    return table_train, table_test

def save_tables(tables, dataset_name, split, data_type='original'):
    for table_name, table in tables.items():
        table_path = f'data/{data_type}/{dataset_name}/{table_name}_{split}.csv'
        if not os.path.exists(os.path.dirname(table_path)):
            os.makedirs(os.path.dirname(table_path))
        table.to_csv(table_path, index=False)

def load_tables(dataset_name, split, data_type='original'):
    tables = {}
    data_path = f'data/{data_type}/{dataset_name}'
    for file_name in os.listdir(data_path):
        table_name = file_name.split(f'_{split}')[0]
        table = pd.read_csv(f'{data_path}/{file_name}')
        tables[table_name] = table
    return tables


def prepare_dataset(dataset_name, seed=42):
    tables = read_original_tables(dataset_name)
    metadata = sdv_metadata.generate_metadata(dataset_name, tables)
    root_table = sdv_metadata.get_root_table(dataset_name)

    tables_train = {table_name: table.copy() for table_name, table in tables.items()}
    tables_test = {table_name: table.copy() for table_name, table in tables.items()}
    train_root, test_root = split_table(tables[root_table], seed=seed)
    tables_train[root_table] = train_root
    tables_test[root_table] = test_root
    tables_train = conditionally_sample(tables_train, metadata, root_table)
    tables_test = conditionally_sample(tables_test, metadata, root_table)
    save_tables(tables_train, dataset_name, 'train')
    save_tables(tables_test, dataset_name, 'test')