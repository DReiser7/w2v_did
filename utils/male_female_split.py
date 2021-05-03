import sys
import csv
import shutil
from pathlib import Path

# python male_female_split.py < file_input.tsv > file_output.csv
base = 'C:/Users/domin/Downloads/cv-corpus-6.1-2020-12-11/de/'
input_file = base + 'validated.tsv'
clips = base + 'clips/'
clips_male = base + 'male/'
clips_female = base + 'female/'

dir_create = Path(clips_male)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_female)
dir_create.mkdir(parents=True, exist_ok=True)

tabin = csv.reader(sys.stdin, dialect=csv.excel_tab)
for row in tabin:
    if row[6] != '' and row[6] != 'gender':
        source = clips + row[1]
        if row[6] == 'male':
            dest = clips_male + row[1]
        if row[6] == 'female':
            dest = clips_female + row[1]
        shutil.copyfile(source, dest)
        print('copying to ' + dest)

print('finished converting!')