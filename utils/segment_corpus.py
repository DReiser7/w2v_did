from pathlib import Path
import soundfile as sf
import librosa
import torchaudio
import numpy as np


def resample_and_segment_audio(dir_path, seconds):
    pathlist = Path(dir_path).glob('**/*.mp3')

    for path in pathlist:
        absolute_path = str(path)
        print('preprocessing ' + absolute_path + ' ...')

        # do resampling on root path
        speech_array, sampling_rate = torchaudio.load(str(path))
        speech_resampled = librosa.resample(np.asarray(speech_array[0].numpy()), sampling_rate, 16_000)

        segmented_path = str(path.parent).replace('test', 'test-segmented-' + str(seconds))

        counter = 0
        for block in sf.blocks(speech_resampled, blocksize=seconds * 16000, overlap=16000, fill_value=0):
            new_file_path = segmented_path + '/' + str(path.name) + '_' + str(counter) + '.mp3'

            filename = Path(segmented_path)
            filename.mkdir(parents=True, exist_ok=True)

            torchaudio.save(new_file_path, block, 16_000, format='mp3')
            counter += 1


if __name__ == "__main__":
    base_dir = '/cluster/home/reisedom/data/spanish-accents-test-aug/test'
    resample_and_segment_audio(base_dir, 1)
