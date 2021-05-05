import pandas as pd

if __name__ == "__main__":

    base_dir = 'C:/workspaces/BA/Corpora/cv-corpus-6.1-2020-12-11/es/'
    test_tsv = base_dir + 'test.tsv'

    list_of_attributes_taken = ['mexicano',
                          'caribe',
                          'andino',
                          'centrosurpeninsular',
                          'americacentral',
                          'rioplatense',
                          'nortepeninsular',
                          'surpeninsular']

    test_data = pd.read_table(test_tsv, sep='\t')
    counter_empty = 0
    counter_taken = 0
    counter_total = 0

    for i in range(0, len(test_data)):
        accent = test_data.iloc[i, 7]
        counter_total = counter_total + 1
        if accent in list_of_attributes_taken:
            counter_taken = counter_taken + 1
        else:
            print(accent)

    print('total: ', str(counter_total))
    print('taken empty: ', str(counter_taken))
    print('empty: ', str(counter_empty))
    print('filled in %: ', str(100 / counter_total * counter_taken))
