import os
import shutil
from pathlib import Path
from random import sample

import pandas as pd


def create_folders(output_dirs, list_of_entries):
    for output_dir in output_dirs:
        for entry in list_of_entries:
            dir_create = Path(output_dir + entry)
            dir_create.mkdir(parents=True, exist_ok=True)


def copy_files(data, output_dir, attributes, column_index):
    for i in range(0, len(data)):
        attribute = data.iloc[i, column_index]

        if attribute in attributes:
            source = clips + data.iloc[i, 1]
            dest = output_dir + attribute + '/' + data.iloc[i, 1]
            shutil.copyfile(source, dest)
            print('copying ' + source + ' to ' + dest)


def reduce_files(output_dir, attributes):
    file_counts_per_attribute = {}
    for attribute in attributes:
        path, dirs, files = next(os.walk(output_dir + attribute))
        file_counts_per_attribute[attribute] = len(files)

    smallest_attribute = min(file_counts_per_attribute, key=file_counts_per_attribute.get)
    number_limit = 2 * file_counts_per_attribute[smallest_attribute]

    print('smallest attribute: ', smallest_attribute)
    print('number limit: ', str(number_limit))

    for attribute in attributes:
        if attribute != smallest_attribute:
            files = os.listdir(output_dir + attribute)
            num_of_deletes = file_counts_per_attribute[attribute] - number_limit
            if num_of_deletes > 0:
                for file in sample(files, num_of_deletes):
                    file_to_remove = output_dir + attribute + '/' + file
                    os.remove(file_to_remove)
                    print('removing file: ', file_to_remove)


if __name__ == "__main__":
    # specific config
    do_copy = False
    do_reduce = True
    base_dir = 'C:/workspaces/BA/Corpora/cv-corpus-6.1-2020-12-11/es/'
    # list_of_attributes = ['austria', 'germany', 'switzerland']
    list_of_attributes = ['mexicano',
                          'caribe',
                          'andino',
                          'centrosurpeninsular',
                          # 'americacentral',
                          'rioplatense',
                          'nortepeninsular',
                          'surpeninsular']
    # list_of_attributes = ['male', 'female']
    # list_of_attributes = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties',
    #                       'nineties']
    # list_of_attributes = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties-nineties']

    attribute_type = 'accent'

    # general config
    tsv_column_indices = {
        'accent': 7,
        'sex': 6,
        'age': 5
    }
    train_tsv = base_dir + 'train.tsv'
    dev_tsv = base_dir + 'dev.tsv'
    test_tsv = base_dir + 'test.tsv'
    clips = base_dir + 'clips/'

    train_dir = base_dir + 'train/'
    test_dir = base_dir + 'test/'

    if do_copy:
        create_folders([train_dir, test_dir], list_of_attributes)

        train_data = pd.read_table(train_tsv, sep='\t')
        dev_data = pd.read_table(dev_tsv, sep='\t')
        test_data = pd.read_table(test_tsv, sep='\t')

        # copy train and dev to train
        copy_files(train_data, train_dir, list_of_attributes, tsv_column_indices[attribute_type])
        copy_files(dev_data, test_dir, list_of_attributes, tsv_column_indices[attribute_type])
        # copy test to test
        copy_files(test_data, test_dir, list_of_attributes, tsv_column_indices[attribute_type])

        print('copying finished!')

    if do_reduce:
        reduce_files(train_dir, list_of_attributes)
        print('reducing finished!')
