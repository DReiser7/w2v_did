from pathlib import Path
import csv

if __name__ == "__main__":

    dir = './data/dev/wav'

    pathlist = Path(dir).glob('**/*.wav')
    folders = []

    with open(dir+'\metadata.csv', 'w', newline='') as csvfile:
        for path in pathlist:
            # because path is object not string
            absolute_path = str(path)
            file_name = str(path.parts[len(path.parts) - 1])
            folder = str(path.parts[len(path.parts) - 2])
            folders.append(folder) if folder not in folders else folders
            label = folders.index(folder)

            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([file_name, label, folder])




