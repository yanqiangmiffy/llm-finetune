#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: utils.py
@time: 2023/11/14
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers


def set_args():
    @dataclass
    class ModelArguments:
        model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

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
                "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
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

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    modelargs, dataargs, trainingargs, = parser.parse_args_into_dataclasses()
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


if __name__ == '__main__':
    model_args, data_args, training_args = set_args()
    print(model_args, data_args, training_args)
