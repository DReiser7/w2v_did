from pathlib import Path
import csv
import soundfile as sf
import librosa


def resample_and_segment_audio(dir, do_segment=False):
    pathlist = Path(dir).glob('**/*.wav')

    for path in pathlist:
        absolute_path = str(path)
        print('preprocessing ' + absolute_path + ' ...')

        # do resampling on root path
        speech, fs = sf.read(path)
        sound = librosa.resample(speech, fs, 16000)
        sf.write(absolute_path, sound, 16000)

        # do segmentation
        if do_segment:
            segmented_path = str(path.parent).replace('wav', 'segmented')
            counter = 0
            for block in sf.blocks(path, blocksize=160000, overlap=16000, fill_value=0):
                new_file_path = segmented_path + '/' + str(path.name) + '_' + str(counter) + '.wav'
                filename = Path(segmented_path)
                filename.mkdir(parents=True, exist_ok=True)

                sf.write(new_file_path, block, 16000)
                counter += 1


def create_metadata_csv(dir):
    pathlist = Path(dir).glob('**/*.wav')
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
    print('start preprocessing ...')
    do_segment = True
    dir_wav = '../data/dev/wav'
    dir_segmented = '../data/dev/segmented'

    resample_and_segment_audio(dir, do_segment)
    create_metadata_csv(dir_wav)
    if do_segment:
        create_metadata_csv(dir_segmented)

    print('... preprocessing done!')
