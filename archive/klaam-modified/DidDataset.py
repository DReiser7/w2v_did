"""Arabic Speech Corpus"""

from __future__ import absolute_import, division, print_function

import os
import random

import datasets
import soundfile as sf
import pandas as pd

_CITATION = ""

_DESCRIPTION = ""


class DialectSpeechCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for DialectSpeechCorpusCorpus."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(DialectSpeechCorpusConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


def map_to_array(batch):
    start, stop = batch['segment'].split('_')
    speech_array, _ = sf.read(batch["file"], start=start, stop=stop)
    batch["speech"] = speech_array
    return batch


class DialectSpeechCorpus(datasets.GeneratorBasedBuilder):
    """DialectSpeechCorpus dataset."""

    BUILDER_CONFIGS = [
        DialectSpeechCorpusConfig(name="clean", description="'Clean' speech."),
    ]

    def _info(self):
        labels_csv = pd.read_csv(self.config.data_files['labels_csv'])
        label_idx = []
        label_names = []
        for i in range(0, len(labels_csv)):
            label_idx.append(labels_csv.iloc[i, 0])
            label_names.append(labels_csv.iloc[i, 1])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(
                        names=label_names
                    )
                }
            ),
            supervised_keys=("file", "text"),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name="train", gen_kwargs={"archive_path": os.path.join(self.config.data_dir, "train")}),
            # datasets.SplitGenerator(name="dev", gen_kwargs={"archive_path": os.path.join(self.config.data_dir, "dev")}),
            datasets.SplitGenerator(name="test", gen_kwargs={"archive_path": os.path.join(self.config.data_dir, "test")}),
        ]

    def _generate_examples(self, archive_path):
        """Generate examples from a Librispeech archive_path."""
        wav_dir = os.path.join(archive_path, "wav")

        paths = []
        labls = []

        for _, c in enumerate(os.listdir(wav_dir)):
            if os.path.isdir(f'{wav_dir}/{c}/'):
                for file in os.listdir(f'{wav_dir}/{c}/')[:2200]:
                    if file.endswith('.wav') or file.endswith('.mp3'):
                        wav_path = f'{wav_dir}/{c}/{file}'
                        paths.append(wav_path)
                        labls.append(c)

        data = list(zip(paths, labls))
        random.Random(4).shuffle(data)
        paths, labls = zip(*data)
        for i in range(len(paths)):
            example = {
                "file": paths[i],
                "label": labls[i]
            }
            yield str(i), example
