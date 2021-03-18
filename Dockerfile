FROM python:3

ADD DidDataset.py /
ADD DidMain.py /
ADD DidModel.py /
ADD DidModelRunner.py /

RUN apt-get update
RUN apt-get --yes install libsndfile1

RUN pip install pandas
RUN pip install soundfile
RUN pip install torch
RUN pip install fairseq

CMD [ "python", "./DidMain.py" ]