import pandas as pd


if __name__ == "__main__":

    base_dir = 'C:/workspaces/BA/Corpora/cv-corpus-6.1-2020-12-11/de/'
    test_tsv = base_dir + 'test.tsv'
    dev_tsv = base_dir + 'dev.tsv'

    accent_set = set()
    test_data = pd.read_table(test_tsv, sep='\t')
    dev_data = pd.read_table(dev_tsv, sep='\t')

    accents = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties',  'nineties']

    count_sex = 'female'

    attributes = ['female', 'male', 'other']

    age_dict = {
        'female': 0,
        'male': 0,
        'other': 0,
        'unknown': 0,
    }


    for i in range(0, len(test_data)):
        accent = test_data.iloc[i, 5]
        attr = test_data.iloc[i, 6]
        if accent in accents and attr in attributes:
            age_dict[attr] = age_dict[attr] + 1
        elif accent in accents:
            age_dict['unknown'] = age_dict['unknown'] + 1

    # for i in range(0, len(dev_data)):
    #     accent = dev_data.iloc[i, 7]
    #     attr = dev_data.iloc[i, 5]
    #     if accent in accents and attr in attributes:
    #         age_dict[attr] = age_dict[attr] + 1
    #     elif accent in accents:
    #         age_dict['unknown'] = age_dict['unknown'] + 1

    for k, v in age_dict.items():
        print(k, str(v))


    print(str(sum(age_dict.values())))