import csv
import sys
from pathlib import Path

import splitfolders


def create_metadata_csv(dir):
    pathlist = Path(dir).glob('**/*.npy')
    folders = []

    print('writing metadata: ', dir + 'metadata.csv')
    with open(dir + 'metadata.csv', 'w', newline='') as csvfile:
        for path in pathlist:
            # because path is object not string
            absolute_path = str(path)
            file_name = str(path.parts[len(path.parts) - 1])
            folder = str(path.parts[len(path.parts) - 2])
            folders.append(folder) if folder not in folders else folders
            label = folders.index(folder)

            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([file_name, label, folder])


if __name__ == "__main__":
    dir_numpy = sys.argv[1]  # example './data/dev/numpy'
    ratio_train = sys.argv[2]
    ratio_val = sys.argv[3]

    output = dir_numpy.replace('numpy', 'splitted')

    path_list = Path(dir_numpy).glob('**/*.npy')

    splitfolders.ratio(dir_numpy, output=output, ratio=(ratio_train, ratio_val), group_prefix=None)  # default values

    create_metadata_csv(output + "/train")
    create_metadata_csv(output + "/val")

    print('... splitting done!')
