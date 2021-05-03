import pandas as pd

train_tsv = 'C:/Users/domin/Downloads/cv-corpus-6.1-2020-12-11/de/train.tsv'
test_tsv = 'C:/Users/domin/Downloads/cv-corpus-6.1-2020-12-11/de/test.tsv'
train = pd.read_table(train_tsv, sep='\t')
test = pd.read_table(test_tsv, sep='\t')

key_set = set()

for i in range(0, len(train)):
    key_set.add(train.iloc[i, 0])

for i in range(0, len(test)):
    if test.iloc[i, 0] in key_set:
        print('duplicate: ', test.iloc[i, 0])

print('check done see duplicates above')
