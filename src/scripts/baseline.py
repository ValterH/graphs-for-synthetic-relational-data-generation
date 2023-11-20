import os
import pickle

import pandas as pd
from rike.generation import sdv_metadata
from sdv.relational import HMA1

from src.data.utils import save_tables, load_tables


def generate_sdv(dataset_name):
    tables_train = load_tables(dataset_name, 'train')
    tables_test = load_tables(dataset_name, 'test')

    metadata = sdv_metadata.generate_metadata(dataset_name, tables_train)
    root_table = sdv_metadata.get_root_table(dataset_name)
    model = HMA1(metadata=metadata)
    model.fit(tables_train)
    model_save_path = f'models/sdv/{dataset_name}/model.pickle'
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    with open(model_save_path, "wb") as f:
        pickle.dump(model, open(model_save_path, "wb" ) )

    synthetic_data_train = model.sample(num_rows=tables_train[root_table].shape[0])
    synthetic_data_test = model.sample(num_rows=tables_test[root_table].shape[0])

    save_tables(synthetic_data_train, dataset_name, 'train', data_type='synthetic/sdv')
    save_tables(synthetic_data_test, dataset_name, 'test', data_type='synthetic/sdv')


if __name__ == '__main__':
    generate_sdv('rossmann-store-sales')
    generate_sdv('mutagenesis')