#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: main.py
@time: 2023/11/14
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import copy
import datetime
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Tuple, List, Dict, Sequence

import pytz
import torch
import transformers
from datasets import load_dataset
from gomall.autolog import log
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer

logger = logging.getLogger(__name__)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )['train']
    return raw_datasets


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: str) -> Dataset:
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(
        data_path=data_path,
    )
    logging.warning("Formatting inputs...")
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer):
        ins_data = examples['instruction']
        input_data = examples['input']
        output = examples['output']

        len_ = len(ins_data)

        sources = [
            prompt_input.format_map({'instruction': ins_data[i], 'input': input_data[i]})
            if
            input_data[i] != ""
            else
            prompt_no_input.format_map({'instruction': ins_data[i]}) for i in range(len_)
        ]
        targets = [
            f"{example}{tokenizer.eos_token}" for example in output]

        input_output = preprocess(
            sources=sources, targets=targets, tokenizer=tokenizer)
        examples['input_ids'] = input_output['input_ids']
        examples['labels'] = input_output['labels']
        return examples

    generate_sources_targets_p = partial(
        generate_sources_targets, tokenizer=tokenizer)

    dataset = dataset.map(
        function=generate_sources_targets_p,
        batched=True,
        desc="Running tokenizer on train dataset",
        num_proc=2
    ).shuffle()
    return dataset


def set_args(
        model_name_or_path="",
        dataset_name_or_path="",
        use_peft=None,
        epochs=None,
        learning_rate=None,
        model_max_length=None,
        split_ratio=None
):
    @dataclass
    class ModelArguments:
        model_name_or_path: Optional[str] = field(default="bigscience/bloom-560m")

    @dataclass
    class DataArguments:
        data_path: str = field(default=None, metadata={
            "help": "Path to the training data."})

    @dataclass
    class TrainingArguments(transformers.TrainingArguments):
        cache_dir: Optional[str] = field(default=None)
        optim: str = field(default="adamw_torch")
        model_max_length: int = field(
            default=512,
            metadata={
                "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
        )
        output_dir: Optional[str] = field(
            default="output",
            metadata={
                "help": "生成模型的目录"},
        )
        use_peft: Optional[bool] = field(
            default=True,
            metadata={
                "help": "is lora fintuning"
            }
        )
        target_modules: Optional[str] = field(default="all")
        lora_rank: Optional[int] = field(default=8)
        lora_dropout: Optional[float] = field(default=0.05)
        lora_alpha: Optional[float] = field(default=32.0)
        modules_to_save: Optional[str] = field(default=None)

    # ModelArguments = ModelArguments(
    #     model_name_or_path=model_name_or_path
    # )
    # TrainingArguments = TrainingArguments(
    #     use_peft=use_peft,
    #     num_train_epochs=epochs,
    #     learning_rate=learning_rate,
    #     model_max_length=model_max_length
    # )

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    modelargs, dataargs, trainingargs, = parser.parse_args_into_dataclasses()

    modelargs.model_name_or_path = model_name_or_path
    dataargs.data_path = dataset_name_or_path
    trainingargs.use_peft = use_peft
    trainingargs.num_train_epochs = epochs
    trainingargs.learning_rate = learning_rate
    trainingargs.model_max_length = model_max_length
    trainingargs.per_device_train_batch_size = 1
    return modelargs, dataargs, trainingargs


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def train(model_args, data_args, training_args):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True if model_args.model_name_or_path.find("falcon") != -1 else False

    )
    torch.cuda.empty_cache()

    if training_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")

        # Set fp32 forward hook for lm_head
        output_layer = getattr(model, "lm_head")
        if isinstance(output_layer, torch.nn.Linear):
            def fp32_forward_pre_hook(module: torch.nn.Module, args: Tuple[torch.Tensor]):
                return args[0].to(output_layer.weight.dtype)

            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)

            output_layer.register_forward_pre_hook(fp32_forward_pre_hook)
            output_layer.register_forward_hook(fp32_forward_post_hook)

        # Load LoRA model
        if training_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {training_args.peft_path}")
            model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            target_modules = training_args.target_modules.split(',') if training_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model)
            modules_to_save = training_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {training_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        logger.info("Fine-tuning method: Full parameters training")

    # Initialize our Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if model_args.model_name_or_path.find("falcon") != -1:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = make_train_dataset(
        tokenizer=tokenizer, data_path=data_args.data_path)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                           label_pad_token_id=IGNORE_INDEX
                                           )

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


def run_task(
        model_name_or_path="bigscience/bloom-560m",
        dataset_name_or_path="",
        use_peft=True,
        epochs=2,
        learning_rate=2e-5,
        model_max_length=2048,
        split_ratio=20,
):
    """

    Args:
        model_name_or_path: 模型路径
        dataset_name_or_path: 数据集路径
        use_peft: 是否使用Lora
        epochs: 迭代轮次
        learning_rate:学习率
        model_max_length:序列长度

    Returns:

    """
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    ## 获取当前时间，并将当前时间作为默认的日志输出文件夹名称
    current_time = datetime.datetime.now(shanghai_tz)
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"llama-7b_fine_tuning21-{formatted_time}"
    log0 = log.AutoLog(name=log_path)
    try:

        model_args, data_args, training_args = set_args(
            model_name_or_path,
            dataset_name_or_path,
            use_peft,
            epochs,
            learning_rate,
            model_max_length,
            split_ratio,
        )
        print(model_args.model_name_or_path, training_args.num_train_epochs)

        log0.log_hyper(
            lr=training_args.learning_rate,
            num_epochs=training_args.num_train_epochs,
            # batch_size=training_args.per_device_train_batch_size,
            lora_rank=training_args.lora_rank
        )
        log0.log_constant("datasets", datasets=data_args.dataset_list[0].dataset_name)
        log0.log_constant("model", model=model_args.model_name_or_path)

        train(model_args, data_args, training_args)
        return {"running_status": "run task successfully", "message": "运行成功", "log_dir": log_path}

    except Exception as e:
        return {"running_status": "run task faild", "message": e, "log_dir": log_path}

if __name__ == "__main__":
    run_task()
