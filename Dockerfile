FROM python:3

ADD DidDataset.py /
ADD DidMain.py /
ADD DidModel.py /
ADD DidModelRunner.py /

RUN pip install torch
RUN pip install soundfile
RUN pip install fairseq


CMD [ "python", "./DidMain.py" ]