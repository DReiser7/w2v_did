#!/usr/bin/env python3
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import librosa
import numpy as np
import torch
import torchaudio
import transformers
from packaging import version
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from transformers import (
    HfArgumentParser,
    Trainer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    is_apex_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import wandb
from processors import CustomWav2Vec2Processor

#######################################################
#            GLOBALS TO MODIFY TRAINING
#######################################################

from models import Wav2VecClassifierModelMean6 as Wav2VecClassifierModel

NUMBER_OF_CLASSES = 6  # has to fit Model!
S_RATE = 16_000
CORPORA_PATH = "corpora/com_voice_english_accent_corpus"
LABEL_IDX = [0, 1, 2, 3, 4, 5]
LABEL_NAMES = [
    'us',
    'australia',
    'canada',
    'england',
    'indian',
    'scotland']

# Parametrized
WINDOW_COUNT = 3
WINDOW_LENGTH = 10
SAMPLE_LENGTH = WINDOW_LENGTH * S_RATE

######################################################

os.environ['WANDB_PROJECT'] = 'w2v_did'
os.environ['WANDB_LOG_MODEL'] = 'true'

if is_apex_available():
    pass

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    device: Optional[str] = field(
        default='cuda', metadata={"help": "The device on which to run)."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    attention_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."},
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
                    "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
                    "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    layerdrop: Optional[float] = field(default=0.0, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train+validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    chars_to_ignore: List[str] = list_field(
        default=[",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�"],
        metadata={"help": "A list of characters to remove from the transcripts."},
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: CustomWav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = SAMPLE_LENGTH
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = SAMPLE_LENGTH
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        input_features = [{"input_values": feature["input_values"]} for feature in features]

        def onehot(lbl):
            onehot = [0] * NUMBER_OF_CLASSES
            onehot[int(lbl)] = 1
            return onehot

        output_features = [onehot(feature["labels"]) for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # for val in batch['input_values']:
        #   print(val[:10])
        #   print(val[-10:])
        # print(batch['input_values'].shape)
        batch["labels"] = torch.tensor(output_features)
        # print(batch["labels"].argmax(-1))
        return batch


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels").to('cuda')
        labels = inputs['labels'].to('cuda')
        outputs = model(**inputs)  # torch.Size([32, 5])
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(outputs['logits'],
                        labels.argmax(-1).long())

        return (loss, outputs) if return_outputs else loss


def main(model_args, data_args, training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets:
    eval_dataset = datasets.load_dataset(CORPORA_PATH, split="test", cache_dir=model_args.cache_dir)

    processor = CustomWav2Vec2Processor.from_pretrained(model_args.model_name_or_path)
    model = Wav2VecClassifierModel.from_pretrained(
        model_args.model_name_or_path,
        attention_dropout=0.01,
        hidden_dropout=0.01,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.01,
        gradient_checkpointing=True,
    )

    if data_args.max_val_samples is not None:
        max_val_samples = min(data_args.max_val_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(max_val_samples))

    # Preprocessing the datasets.
    # We need to read the aduio files as arrays and tokenize the targets.
    def speech_file_to_array_fn(batch, start_param, stop_param):
        speech_array, sampling_rate = torchaudio.load(batch["file"])
        speech_array = speech_array[0].numpy()[(start_param * sampling_rate):(stop_param * sampling_rate)]
        batch["speech"] = librosa.resample(np.asarray(speech_array), sampling_rate, S_RATE)
        batch["sampling_rate"] = S_RATE
        batch["parent"] = batch["label"]
        return batch

    def filter_null(batch):
        return not (batch['speech'] == np.array([0])).all()

    eval_dataset_array = []
    stop = 0
    for i in range(WINDOW_COUNT):
        start = 0 if i == 0 else stop
        stop = start + WINDOW_LENGTH
        arguments = {'start_param': start, 'stop_param': stop}
        eval_dataset_array.append(eval_dataset.map(
            speech_file_to_array_fn,
            remove_columns=eval_dataset.column_names,
            num_proc=data_args.preprocessing_num_workers,
            fn_kwargs=arguments

        ).filter(filter_null))

    eval_dataset = datasets.concatenate_datasets(eval_dataset_array)

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
                len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        batch["labels"] = batch["parent"]
        return batch

    eval_dataset = eval_dataset.map(
        prepare_dataset,
        remove_columns=eval_dataset.column_names,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )

    from sklearn.metrics import classification_report, confusion_matrix

    def compute_metrics(pred):
        label_idx = LABEL_IDX
        label_names = LABEL_NAMES
        labels = pred.label_ids.argmax(-1)
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        report = classification_report(y_true=labels, y_pred=preds, labels=label_idx, target_names=label_names)
        matrix = confusion_matrix(y_true=labels, y_pred=preds)
        print(report)
        print(matrix)

        wandb.log(
            {"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds, class_names=label_names)})

        wandb.log(
            {"precision_recall": wandb.plot.pr_curve(y_true=labels, y_probas=pred.predictions, labels=label_names)})

        return {"accuracy": acc, "f1_score": f1}

    wandb.init(name=training_args.output_dir, config=training_args)

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Initialize our Trainer
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results


if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    WINDOW_LENGTH = int(sys.argv[2])
    WINDOW_COUNT = math.ceil(10 / WINDOW_LENGTH)
    SAMPLE_LENGTH = WINDOW_LENGTH * S_RATE
    print(str(WINDOW_LENGTH))
    print(str(WINDOW_COUNT))

    training_args.output_dir = training_args.output_dir + str(WINDOW_LENGTH)
    main(model_args=model_args, data_args=data_args, training_args=training_args)
