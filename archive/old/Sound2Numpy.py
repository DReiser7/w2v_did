import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def create_metadata_csv(dir):
    pathlist = Path(dir).glob('**/*.npy')
    folders = []

    print('writing metadata: ', dir + '\metadata.csv')
    with open(dir + '\metadata.csv', 'w', newline='') as csvfile:
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
    dir_sound = sys.argv[1]  # example './data/dev/segmented'
    path_list = Path(dir_sound).glob('**/*.wav')

    for path in path_list:
        absolute_path = str(path)
        print('sound2Numpy ' + absolute_path + ' ...')

        # change to numpy array
        speech, fs = sf.read(path)
        np_array = np.array(speech)

        numpy_path = str(path.parent).replace('segmented', 'numpy')
        new_file_path = numpy_path + '/' + str(path.name) + '.npy'

        filename = Path(numpy_path)
        filename.mkdir(parents=True, exist_ok=True)

        np.save(new_file_path, np_array)

    create_metadata_csv(dir_sound.replace('segmented', 'numpy'))

    print('... preprocessing done!')
