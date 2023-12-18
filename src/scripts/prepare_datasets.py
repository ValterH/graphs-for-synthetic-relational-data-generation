import os
import shutil

from tabsyn.process_dataset import process_data
from src.data.utils import prepare_dataset, load_metadata


def main():
    prepare_dataset('rossmann-store-sales')
    prepare_dataset('mutagenesis')

    # prepare tabsyn data
    rossmann_metadata = load_metadata('rossmann-store-sales')
    mutagenesis_metadata = load_metadata('mutagenesis')
    for table in rossmann_metadata.get_tables():
        os.makedirs(f'tabsyn/data/{table}', exist_ok=True)
        shutil.copy(f'data/original/rossmann-store-sales/{table}_train.csv', f'tabsyn/data/{table}/{table}_train.csv')
        shutil.copy(f'data/original/rossmann-store-sales/{table}_test.csv', f'tabsyn/data/{table}/{table}_test.csv')


    for table in mutagenesis_metadata.get_tables():
        os.makedirs(f'tabsyn/data/{table}', exist_ok=True)
        shutil.copy(f'data/original/mutagenesis/{table}_train.csv', f'tabsyn/data/{table}/{table}_train.csv')
        shutil.copy(f'data/original/mutagenesis/{table}_test.csv', f'tabsyn/data/{table}/{table}_test.csv')

    os.chdir('tabsyn')
    for table in rossmann_metadata.get_tables():
        process_data(table)
    
    for table in mutagenesis_metadata.get_tables():
        process_data(table)


if __name__ == '__main__':
    main()