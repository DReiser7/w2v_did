FROM python:3

ADD DidDataset.py /
ADD DidMain.py /
ADD DidModel.py /
ADD DidModelRunner.py /
ADD DidModelHuggingFace.py /

RUN apt-get update
RUN apt-get --yes install libsndfile1
RUN apt --yes install git-all

RUN pip install pandas
RUN pip install soundfile
RUN pip install torch
RUN pip install transformers
RUN pip install wandb
RUN pip install split-folders

#install fairseq over repo
RUN git clone https://github.com/pytorch/fairseq
RUN pip install --editable ./fairseq/

RUN git clone https://github.com/DReiser7/w2v_did.git
RUN cd w2v_did
RUN git checkout wandb

#CMD [ "python", "./DidMain.py" ]
CMD ["sh", "-c", "python ./DidMain.py  $TRAIN $TEST $MODEL $EPOCHS $BSIZE"]
#CMD python DidMain.py $TRAIN $TEST $MODEL