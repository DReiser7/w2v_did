import sys
from pathlib import Path
from pydub import AudioSegment
import multiprocessing as mp


def convert_file(path, input_dir, output_dir, input_format, output_format):
    print('converting ' + str(path) + ' from ' + input_format + ' to ' + output_format)
    file = AudioSegment.from_file(path)
    subdir = str(path.parent).replace('\\', '/').replace(input_dir, '')
    path_file = output_dir + subdir + '/' + path.stem + '.' + output_format

    dir_create = Path(output_dir + subdir)
    dir_create.mkdir(parents=True, exist_ok=True)

    file.export(path_file, format=output_format)


# python AudioConverter.py "/input" "/output" "m4a" "mp3"
if __name__ == "__main__":
    input_dir = sys.argv[1]  # '/input'
    output_dir = sys.argv[2]  # '/output'
    input_format = sys.argv[3]  # 'm4a'
    output_format = sys.argv[4]  # 'mp3'

    path_list = Path(input_dir).glob('**/*.' + input_format)

    print('parallelizing level: ' + str(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count())
    pool.starmap_async(convert_file, [(path, input_dir, output_dir, input_format, output_format) for path in path_list])

    pool.close()
    pool.join()

    print('finished converting!')
