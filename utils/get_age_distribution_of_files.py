import pandas as pd

train_tsv = 'C:/workspaces/BA/Corpora/cv-corpus-6.1-2020-12-11/de/train.tsv'
test_tsv = 'C:/workspaces/BA/Corpora/cv-corpus-6.1-2020-12-11/de/test.tsv'
dev_tsv = 'C:/workspaces/BA/Corpora/cv-corpus-6.1-2020-12-11/de/dev.tsv'

files_to_check = 'C:/workspaces/BA/Corpora/4000_3.csv'

train = pd.read_table(train_tsv, sep='\t')
test = pd.read_table(test_tsv, sep='\t')
dev = pd.read_table(dev_tsv, sep='\t')

files = pd.read_table(files_to_check, sep=',')

attributes = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties',
              'nineties']

age_dict_female = {
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

age_dict_male = {
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


def increment_dict(sex, age, file_name):
    if sex == 'female':
        if age in attributes:
            age_dict_female[age] = age_dict_female[age] + 1
        else:
            age_dict_female['unknown'] = age_dict_female['unknown'] + 1
    else:
        if age in attributes:
            age_dict_male[age] = age_dict_male[age] + 1
        else:
            age_dict_male['unknown'] = age_dict_male['unknown'] + 1


for i in range(0, len(files)):
    path = files.iloc[i, 2]
    path_parts = path.split(sep="/")
    file_name = path_parts[len(path_parts) - 1]
    sex = path_parts[len(path_parts) - 2]

    # for x in range(0, len(train)):
    #     if train.iloc[x, 1] == file_name:
    #         age = train.iloc[x, 5]
    #         increment_dict(sex, age, file_name)
    #         print("file found with age: " + str(age))
    #         break

    for y in range(0, len(test)):
        if test.iloc[y, 1] == file_name:
            age = test.iloc[y, 5]
            increment_dict(sex, age, file_name)
            # print("file found with age: " + str(age))
            print("file found with id: " + test.iloc[y, 0])
            break

    # for z in range(0, len(dev)):
    #     if dev.iloc[z, 1] == file_name:
    #         age = dev.iloc[z, 5]
    #         increment_dict(sex, age, file_name)
    #         print("file found with age: " + str(age))
    #         break

print("results female")
for k, v in age_dict_female.items():
    print(k, str(v))

print("results male")
for k, v in age_dict_male.items():
    print(k, str(v))