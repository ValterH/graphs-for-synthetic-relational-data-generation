import os
import pickle

from sdv.relational import HMA1

from src.data.utils import save_tables, load_tables, load_metadata, get_root_table


def generate_sdv(dataset_name):
    tables_train = load_tables(dataset_name, split='train')
    metadata = load_metadata(dataset_name)
    root_table = get_root_table(dataset_name)
    model = HMA1(metadata=metadata)
    model.fit(tables_train)
    model_save_path = f'models/sdv/{dataset_name}/model.pickle'
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    with open(model_save_path, "wb") as f:
        pickle.dump(model, open(model_save_path, "wb" ) )

    synthetic_data = model.sample(num_rows=tables_train[root_table].shape[0])

    save_tables(synthetic_data, dataset_name, data_type='synthetic/sdv')


if __name__ == '__main__':
    generate_sdv('rossmann-store-sales')
    generate_sdv('mutagenesis')