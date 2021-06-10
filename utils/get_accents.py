import pandas as pd


if __name__ == "__main__":

    base_dir = 'C:/workspaces/BA/Corpora/cv-corpus-6.1-2020-12-11/es/'
    test_tsv = base_dir + 'test.tsv'

    accent_set = set()
    test_data = pd.read_table(test_tsv, sep='\t')

    for i in range(0, len(test_data)):
        accent = test_data.iloc[i, 7]
        if accent not in accent_set:
            accent_set.add(accent)
            print(accent)