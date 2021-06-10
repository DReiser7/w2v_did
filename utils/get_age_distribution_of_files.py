import pandas as pd

train_tsv = 'C:/Users/domin/Downloads/cv-corpus-6.1-2020-12-11/en/train.tsv'
test_tsv = 'C:/Users/domin/Downloads/cv-corpus-6.1-2020-12-11/en/test.tsv'
dev_tsv = 'C:/Users/domin/Downloads/cv-corpus-6.1-2020-12-11/en/dev.tsv'

files_to_check = 'C:/workspaces/BA/Corpora/eval_english_5000_2.csv'

train = pd.read_table(train_tsv, sep='\t')
test = pd.read_table(test_tsv, sep='\t')
dev = pd.read_table(dev_tsv, sep='\t')

files = pd.read_table(files_to_check, sep=',')


attributes = ['female', 'male', 'other']

dict_attribute = {
    'female': 0,
    'male': 0,
    'other': 0,
    'unknown': 0,
}

def increment_dict(age):
    if age in attributes:
        dict_attribute[age] = dict_attribute[age] + 1
    else:
        dict_attribute['unknown'] = dict_attribute['unknown'] + 1


for i in range(0, len(files)):
    path = files.iloc[i, 2]
    path_parts = path.split(sep="/")
    file_name = path_parts[len(path_parts) - 1]
    clazz = path_parts[len(path_parts) - 2]

    idx = 6
    attribute = 'sex'

    # for x in range(0, len(train)):
    #     if train.iloc[x, 1] == file_name:
    #         age = train.iloc[x, 5]
    #         increment_dict(sex, age, file_name)
    #         print("file found with age: " + str(age))
    #         break

    # files.loc[files.index[i], 'correct'] = clazz

    for y in range(0, len(test)):
        if test.iloc[y, 1] == file_name:
            age = test.iloc[y, idx]
            increment_dict(age)
            print("file found with age: " + str(age))
            files.loc[files.index[i], attribute] = age
            break

    for z in range(0, len(dev)):
        if dev.iloc[z, 1] == file_name:
            age = dev.iloc[z, idx]
            increment_dict( age)
            print("file found with age: " + str(age))
            files.loc[files.index[i], attribute] = age
            break


files.to_csv(files_to_check, index=False)
print("results")
for k, v in dict_attribute.items():
    print(k, str(v))