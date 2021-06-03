import pandas as pd


if __name__ == "__main__":

    base_dir = 'C:/workspaces/BA/Corpora/cv-corpus-6.1-2020-12-11/de/'
    test_tsv = base_dir + 'train.tsv'
    dev_tsv = base_dir + 'dev.tsv'

    accent_set = set()
    test_data = pd.read_table(test_tsv, sep='\t')
    dev_data = pd.read_table(dev_tsv, sep='\t')

    counter = 0

    count_sex = 'female'

    attributes = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties',
                  'nineties']

    age_dict = {
        'teens': 0,
        'twenties': 0,
        'thirties': 0,
        'fourties': 0,
        'fifties': 0,
        'sixties': 0,
        'seventies': 0,
        'eighties': 0,
        'nineties': 0,
        'unknown': 0,
    }


    for i in range(0, len(test_data)):
        sex = test_data.iloc[i, 6]
        age = test_data.iloc[i, 5]
        if age in attributes:
            age_dict[age] = age_dict[age] + 1
        # elif sex == count_sex:
        #     age_dict['unknown'] = age_dict['unknown'] + 1

    for i in range(0, len(dev_data)):
        sex = dev_data.iloc[i, 6]
        age = dev_data.iloc[i, 5]
        if  age in attributes:
            age_dict[age] = age_dict[age] + 1
        # elif sex == count_sex:
        #     age_dict['unknown'] = age_dict['unknown'] + 1

    for k, v in age_dict.items():
        print(k, str(v))


    print(str(sum(age_dict.values())))