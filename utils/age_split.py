import sys
import csv
import shutil
from pathlib import Path

# python male_female_split.py < file_input.tsv > file_output.csv
base = 'C:/Users/domin/Downloads/cv-corpus-6.1-2020-12-11/de/'
input_file = base + 'validated.tsv'
clips = base + 'clips/'
clips_teens = base + 'teens/'
clips_twenties = base + 'twenties/'
clips_thirties = base + 'thirties/'
clips_fourties = base + 'fourties/'
clips_fifties = base + 'fifties/'
clips_sixties = base + 'sixties/'
clips_seventies = base + 'seventies/'
clips_eighties = base + 'eighties/'
clips_nineties = base + 'nineties/'

dir_create = Path(clips_teens)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_twenties)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_thirties)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_fourties)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_fifties)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_sixties)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_seventies)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_eighties)
dir_create.mkdir(parents=True, exist_ok=True)

dir_create = Path(clips_nineties)
dir_create.mkdir(parents=True, exist_ok=True)

tabin = csv.reader(sys.stdin, dialect=csv.excel_tab)
for row in tabin:
        if row[5] != '' and row[5] != 'age':
            source = clips + row[1]
            if row[5] == 'teens':
                dest = clips_teens + row[1]
            if row[5] == 'twenties':
                dest = clips_twenties + row[1]
            if row[5] == 'thirties':
                dest = clips_thirties + row[1]
            if row[5] == 'fourties':
                dest = clips_fourties + row[1]
            if row[5] == 'fifties':
                dest = clips_fifties + row[1]
            if row[5] == 'sixties':
                dest = clips_sixties + row[1]
            if row[5] == 'seventies':
                dest = clips_seventies + row[1]
            if row[5] == 'eighties':
                dest = clips_eighties + row[1]
            if row[5] == 'nineties':
                dest = clips_nineties + row[1]
            shutil.copyfile(source, dest)
            print('copying to ' + dest)

print('finished converting!')