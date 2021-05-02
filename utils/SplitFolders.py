import csv
import sys
from pathlib import Path

import splitfolders

if __name__ == "__main__":
    input_dir = sys.argv[1]  # example './data/dev/numpy'
    ratio_train = float(sys.argv[2])
    ratio_val = float(sys.argv[3])

    output = input_dir + '-splitted'

    splitfolders.ratio(input_dir, output=output, ratio=(ratio_train, ratio_val), group_prefix=None)  # default values

    print('... splitting done!')
