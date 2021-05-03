import sys
import csv
import shutil
from pathlib import Path

# python male_female_split.py < file_input.tsv > file_output.csv
base = 'C:/Users/domin/Downloads/cv-corpus-6.1-2020-12-11/de/'
input_file = base + 'validated.tsv'
clips = base + 'clips/'
clips_switzerland = base + 'switzerland/'
clips_austria = base + 'austria/'
clips_germany = base + 'germany/'

dir_create = Path(clips_switzerland)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_austria)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_germany)
dir_create.mkdir(parents=True, exist_ok=True)

tabin = csv.reader(sys.stdin, dialect=csv.excel_tab)
for row in tabin:
    if row[7] != '' and row[7] != 'accent':
        source = clips + row[1]
        dest = ''
        if row[7] == 'switzerland':
            dest = clips_switzerland + row[1]
        if row[7] == 'austria':
            dest = clips_austria + row[1]
        if row[7] == 'germany':
            dest = clips_germany + row[1]

        if dest != '':
            shutil.copyfile(source, dest)
            print('copying to ' + dest)

print('finished converting!')