import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from dataset import load_dataset
from trainer_all_samples import PtrTrainerPredictit

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
)

from modeling_bert import BertMultiSampelsWithHingeLoss

from transformers.trainer_utils import is_main_process

import os.path
import json

logger = logging.getLogger(__name__)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    model_choice: Optional[str] = field(default="normal")

    finished_epoch: int = field(
        default=0,
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
                    "the score of the null answer minus this threshold, the null answer is selected for this example. "
                    "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    predict_da: bool = field(
        default=False,
    )

    task: str = field(
        default="sbv",
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
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
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.num_labels = 1
    training_args.num_labels = 1
    # negative_samples = 6

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        eos_token="[eos]",
        pad_token="[PAD]",
    )

    model = BertMultiSampelsWithHingeLoss.from_pretrained(
        model_args.model_name_or_path,
        loss_type="infoNce",
        do_predict=training_args.do_predict,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.num_labels = 1

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    datasets = load_dataset()


    def prepare_features_100samples(examples):
        query_tokens = tokenizer(
            examples["query"],
        )
        knowledge_tokens = tokenizer(
            examples["knowledge"],
        )
        response_tokens = tokenizer(
            examples["response"]
        )

        max_length = data_args.max_seq_length

        query_input_ids = query_tokens.pop("input_ids")   ## [SEP] token id is 102
        knowledge_input_ids = knowledge_tokens.pop("input_ids")
        response_input_ids = response_tokens.pop("input_ids")
        
        query_attention_mask = [[1] * len(c) + [0] * max_length - len(c) for c in query_input_ids]
        query_token_type_ids = [[0] * len(k) + [1] * (len(c) - 1) + [0] * (max_length - len(c) - len(k) + 1) for c, k in
                          zip(query_input_ids, knowledge_input_ids)]


        examples["input_ids"] = input_ids
        examples["attention_mask"] = attention_mask
        examples["token_type_ids"] = token_type_ids
        return examples

    def prepare_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        context_tokens = tokenizer(
            examples["context"],
        )
        knowledge_tokens = tokenizer(
            examples["knowledge"]
        )
        max_length = data_args.max_seq_length
        context_input_ids = context_tokens.pop("input_ids")
        knowledge_input_ids = knowledge_tokens.pop("input_ids")
        input_ids = [k + c[1:] + [0] * (max_length - len(c) - len(k) + 1) for c, k in
                     zip(context_input_ids, knowledge_input_ids)]
        attention_mask = [[1] * (len(c) + len(k) - 1) + [0] * (max_length - len(c) - len(k) + 1) for c, k in
                          zip(context_input_ids, knowledge_input_ids)]
        token_type_ids = [[0] * len(k) + [1] * (len(c) - 1) + [0] * (max_length - len(c) - len(k) + 1) for c, k in
                          zip(context_input_ids, knowledge_input_ids)]


        labels = examples["label"]
        # tokenized_examples["labels"] = labels
        examples_ids = examples["id"]
        ids2_group = {}
        for id_index, id in enumerate(examples_ids):
            if id not in ids2_group.keys():
                ids2_group[id] = [id_index]
            else:
                ids2_group[id].append(id_index)

        examples = {}

        for id, group in ids2_group.items():
            pos_num = 0
            nag_num = 0
            if len(group) != GROUP_EXAMPLES:
                continue
            flag = False
            for example_index in group:
                if len(input_ids[example_index]) > max_length:
                    flag = True
                    break
            if flag is True:
                print("skip!")
                continue

            for example_index in group:
                if labels[example_index] == 1:
                    if "positive_{}_{}".format("input_ids", pos_num) not in examples.keys():
                        examples["positive_{}_{}".format("input_ids", pos_num)] = [input_ids[example_index]]
                    else:
                        examples["positive_{}_{}".format("input_ids", pos_num)].append(input_ids[example_index])

                    if "positive_{}_{}".format("attention_mask", pos_num) not in examples.keys():
                        examples["positive_{}_{}".format("attention_mask", pos_num)] = [attention_mask[example_index]]
                    else:
                        examples["positive_{}_{}".format("attention_mask", pos_num)].append(
                            attention_mask[example_index])

                    if "positive_{}_{}".format("token_type_ids", pos_num) not in examples.keys():
                        examples["positive_{}_{}".format("token_type_ids", pos_num)] = [token_type_ids[example_index]]
                    else:
                        examples["positive_{}_{}".format("token_type_ids", pos_num)].append(
                            token_type_ids[example_index])
                    pos_num += 1
                else:
                    if "nagative_{}_{}".format("input_ids", nag_num) not in examples.keys():
                        examples["nagative_{}_{}".format("input_ids", nag_num)] = [input_ids[example_index]]
                    else:
                        examples["nagative_{}_{}".format("input_ids", nag_num)].append(input_ids[example_index])

                    if "nagative_{}_{}".format("attention_mask", nag_num) not in examples.keys():
                        examples["nagative_{}_{}".format("attention_mask", nag_num)] = [attention_mask[example_index]]
                    else:
                        examples["nagative_{}_{}".format("attention_mask", nag_num)].append(
                            attention_mask[example_index])

                    if "nagative_{}_{}".format("token_type_ids", nag_num) not in examples.keys():
                        examples["nagative_{}_{}".format("token_type_ids", nag_num)] = [token_type_ids[example_index]]
                    else:
                        examples["nagative_{}_{}".format("token_type_ids", nag_num)].append(
                            token_type_ids[example_index])
                    nag_num += 1
            # if len(tmp_nagative.keys()) != len(tmp_positive.keys()):
            #     continue
            # positive.append(tmp_positive)
            # nagative.append(tmp_nagative)


        return examples

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            remove_columns=datasets["train"].column_names
        )
        # assert len(train_dataset["input_ids"]) == len(train_dataset["labels"])

    if training_args.do_eval:
        validation_dataset = datasets["validation"].map(
            prepare_features_100samples,
            batched=True,
            remove_columns=datasets["validation"].column_names
        )
        # assert len(validation_dataset["input_ids"]) == len(validation_dataset["labels"])

    if training_args.do_predict:
        prediction_dataset = datasets["test"].map(
            prepare_features_100samples,
            batched=True,
            remove_columns=datasets["test"].column_names
        )
        # assert len(prediction_dataset["input_ids"]) == len(prediction_dataset["labels"])


    def compute_metrics_for_predict(p: EvalPrediction, test_examples):
        group = {}
        knowledge_group = {}
        prediction = p.predictions.reshape(-1)
        ids_ = test_examples["id"]
        examples = test_examples["knowledge"]
        assert len(ids_) == prediction.shape[0]
        for cnt, id in enumerate(ids_):
            if id not in group.keys():
                group[id] = []
                knowledge_group[id] = []
                group[id].append(prediction[cnt])
                knowledge_group[id].append(examples[cnt])
            else:
                group[id].append(prediction[cnt])
                knowledge_group[id].append(examples[cnt])
        max_list = []
        for id, scores in group.items():
            min_sort_id = np.array(scores).argsort()
            max_sort_id = np.flipud(min_sort_id)
            max_list.append(max_sort_id)

        cnt = 0
        cnt_2 = 0
        cnt_3 = 0
        cnt_5 = 0

        for idx in max_list:
            if 0 in idx[:1]:
                cnt += 1
        for idx in max_list:
            if len(idx) >= 2:
                if 0 in idx[:2]:
                    cnt_2 += 1
        for idx in max_list:
            if len(idx) >= 3:
                if 0 in idx[:3]:
                    cnt_3 += 1
        for idx in max_list:
            if len(idx) >= 5:
                if 0 in idx[:5]:
                    cnt_5 += 1

        return {"accuracy": float(cnt / len(max_list)),
                "recall_2": float(cnt_2 / len(max_list)),
                "recall_3": float(cnt_3 / len(max_list)),
                "recall_5": float(cnt_5 / len(max_list)),
                }

    # Initialize our Trainer
    training_args.remove_unused_columns = False
    training_args.use_legacy_prediction_loop = True

    trainer = PtrTrainerPredictit(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_for_predict,
    )

    ##Test

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

    elif training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=validation_dataset)

        output_eval_file = os.path.join(training_args.output_dir, "predict_results.json")
        with open(output_eval_file, "w") as fp:
            json.dump(metrics, fp)
        output_eval_file = os.path.join(training_args.output_dir, "predict_results_unseen.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Prediction results *****")
            for key, value in sorted(metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    else:
        metrics = trainer.predict(test_dataset=prediction_dataset, test_examples=datasets["test"])
        output_eval_file = os.path.join(training_args.output_dir, "predict_results.json")
        with open(output_eval_file, "w") as fp:
            json.dump(metrics, fp)
        output_eval_file = os.path.join(training_args.output_dir, "predict_results_unseen.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Prediction results *****")
            for key, value in sorted(metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()

