# -*- coding: utf-8 -*-
import math
import os
from dataclasses import dataclass, field
from glob import glob
from types import MethodType
from typing import Literal, Optional, Tuple, List, Dict, Sequence

import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    BloomForCausalLM,
    AutoModel,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    BloomTokenizerFast,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError:
    print("FlashAttention-2 is not installed, ignore this if you are not using FlashAttention.")

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings."}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    shift_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."}
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={"help": "The alpha parameter to control the noise magnitude in NEFTune. value can be 5."}
    )

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError(
                "You must specify a valid model_type to run training. Available model types are " + ", ".join(
                    MODEL_CLASSES.keys()))
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."})
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The prompt template name."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.max_train_samples is not None and 0 < self.max_train_samples <= 1000:
            logger.warning("You may set max_train_samples = -1 to run all samples in production.")


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None, metadata={"help": "The path to the peft model"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum model context length. suggest: 8192 * 4, 8192 * 2, 8192, 4096, 2048, 1024, 512"}
    )

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("You must specify a valid model_max_length >= 60 to run training")


# Copied from: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/extras/patches/llama_patch.py
class LlamaShiftShortAttention(LlamaAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:  # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if getattr(self, "num_key_value_groups"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
            num_groups = q_len // groupsz

            def shift(state: torch.Tensor) -> torch.Tensor:
                state = state.transpose(1, 2)  # output: (bsz, seq_len, n_heads, head_dim)
                state = torch.cat((
                    state[:, :, :self.num_heads // 2], state[:, :, self.num_heads // 2:].roll(-groupsz // 2, dims=1)
                ), dim=2)
                return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim).transpose(1, 2)

            query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, :groupsz, :groupsz].repeat(num_groups, 1, 1, 1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)  # (bsz, :, seq_len, :) or (bsz*n_group, :, groupsz, :)
        attn_output = attn_output.transpose(1, 2).contiguous()

        if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
            attn_output = torch.cat((
                attn_output[:, :, :self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2:].roll(groupsz // 2, dims=1)
            ))

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # FlashAttention requires the input to have the shape (bsz, seq_len, n_heads, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:  # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # cast to half precision
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning("The input hidden states seems to be silently casted in float32.")
            query_states = query_states.to(self.config.torch_dtype)
            key_states = key_states.to(self.config.torch_dtype)
            value_states = value_states.to(self.config.torch_dtype)

        if getattr(self, "num_key_value_groups", None):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)
        key_states = key_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)
        value_states = value_states.transpose(1, 2)  # (bsz, seq_len, n_heads, head_dim)

        if getattr(self.config, "group_size_ratio", None) and self.training:  # shift
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            assert q_len % groupsz == 0, "q_len {} should be divisible by group size {}.".format(q_len, groupsz)
            num_groups = q_len // groupsz

            def shift(state: torch.Tensor) -> torch.Tensor:
                state = torch.cat((
                    state[:, :, :self.num_heads // 2], state[:, :, self.num_heads // 2:].roll(-groupsz // 2, dims=1)
                ), dim=2)
                return state.reshape(bsz * num_groups, groupsz, self.num_heads, self.head_dim)

            query_states, key_states, value_states = shift(query_states), shift(key_states), shift(value_states)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(bsz * num_groups, groupsz)

        if attention_mask is not None:
            logger.warning("Padded sequences are less efficient in FlashAttention.")
            # -q_len: assumes left padding when q_len != kv_len
            unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_states, attention_mask[:, -q_len:])
            unpadded_k, _, cu_seqlens_k, max_seqlen_k = unpad_input(key_states, attention_mask)
            unpadded_v, _, _, _ = unpad_input(value_states, attention_mask)
            attn_output_unpad = flash_attn_varlen_func(
                unpadded_q,
                unpadded_k,
                unpadded_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=True,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, bsz, q_len)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, 0.0, softmax_scale=None, causal=True
            )

        if getattr(self.config, "group_size_ratio", None) and self.training:  # shift back
            groupsz = int(q_len * getattr(self.config, "group_size_ratio"))
            attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
            attn_output = torch.cat((
                attn_output[:, :, :self.num_heads // 2],
                attn_output[:, :, self.num_heads // 2:].roll(groupsz // 2, dims=1)
            ))

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: torch.Tensor,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int
) -> torch.Tensor:
    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


@dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system_prompt: str
    # All messages. format: list of [question, answer]
    messages: Optional[List[Sequence[str]]]
    # The roles of the speakers
    roles: Optional[Sequence[str]]
    # Conversation prompt
    prompt: str
    # Separator
    sep: str
    # Stop token, default is tokenizer.eos_token
    stop_str: Optional[str] = "</s>"

    def get_prompt(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> str:
        """
        Returns a string containing prompt without response.
        """
        return "".join(self._format_example(messages, system_prompt))

    def get_dialog(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        """
        Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.
        """
        return self._format_example(messages, system_prompt)

    def _format_example(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        system_prompt = system_prompt or self.system_prompt
        system_prompt = system_prompt + self.sep if system_prompt else ""  # add separator for non-empty system prompt
        messages = messages or self.messages
        convs = []
        for turn_idx, [user_query, bot_resp] in enumerate(messages):
            if turn_idx == 0:
                convs.append(system_prompt + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs

    def append_message(self, query: str, answer: str):
        """Append a new message."""
        self.messages.append([query, answer])


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation):
    """Register a new conversation template."""
    conv_templates[template.name] = template


"""Vicuna v1.1 template
Supports: https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
          https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
"""
register_conv_template(
    Conversation(
        name="vicuna",
        system_prompt="A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        messages=[],
        roles=("USER", "ASSISTANT"),
        prompt="USER: {query} ASSISTANT:",
        sep="</s>",
    )
)

"""Alpaca template"""
register_conv_template(
    Conversation(
        name="alpaca",
        system_prompt="Below is an instruction that describes a task. "
                      "Write a response that appropriately completes the request.",
        messages=[],
        roles=("### Instruction", "### Response"),
        prompt="### Instruction:\n{query}\n\n### Response:\n",
        sep="\n\n",
    )
)

"""Baichuan template
source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_utils.py#L31
Support: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan",
        system_prompt="",
        messages=[],
        roles=("<reserved_102>", "<reserved_103>"),
        prompt="<reserved_102>{query}<reserved_103>",
        sep="</s>",
    )
)

"""Baichuan2 template
Support: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
         https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat
"""
register_conv_template(
    Conversation(
        name="baichuan2",
        system_prompt="",
        messages=[],
        roles=("<reserved_106>", "<reserved_107>"),
        prompt="<reserved_106>{query}<reserved_107>",
        sep="</s>",
    )
)

"""ziya template"""
register_conv_template(
    Conversation(
        name="ziya",
        system_prompt="",
        messages=[],
        roles=("<human>", "<bot>"),
        prompt="<human>:{query}\n<bot>:",
        sep="\n",
    )
)

"""Linly template"""
register_conv_template(
    Conversation(
        name="linly",
        system_prompt="",
        messages=[],
        roles=("User", "Bot"),
        prompt="User: {query}\nBot: ",
        sep="\n",
    )
)

"""ChatGLM1 template
Support: https://huggingface.co/THUDM/chatglm-6b
source: https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1307
"""
register_conv_template(
    Conversation(
        name="chatglm",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n答：",
        sep="\n",
    )
)

"""ChatGLM2 template
Support: https://huggingface.co/THUDM/chatglm2-6b
source: https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1007
"""
register_conv_template(
    Conversation(
        name="chatglm2",
        system_prompt="",
        messages=[],
        roles=("问", "答"),
        prompt="问：{query}\n\n答：",
        sep="\n\n",
    )
)

"""ChatGLM3 template
Support: https://huggingface.co/THUDM/chatglm3-6b
source: https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py#L179
"""
register_conv_template(
    Conversation(
        name="chatglm3",
        system_prompt="",
        messages=[],
        roles=("<|user|>", "<|assistant|>"),
        prompt="<|user|>\n{query}<|assistant|>",
        sep="\n",
        stop_str="<|user|>",
    )
)

"""Phoenix template"""
register_conv_template(
    Conversation(
        name="phoenix",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: <s>{query}</s>Assistant: ",
        sep="</s>",
    )
)

"""belle template
Supports: https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B
"""
register_conv_template(
    Conversation(
        name="belle",
        system_prompt="",
        messages=[],
        roles=("Human", "Belle"),
        prompt="Human: {query}\n\nBelle: ",
        sep="\n\n",
    )
)

"""aquila template
Supports: https://huggingface.co/qhduan/aquilachat-7b
          https://huggingface.co/BAAI/AquilaChat2-34B
"""
register_conv_template(
    Conversation(
        name="aquila",
        system_prompt="A chat between a curious human and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}###Assistant:",
        sep="###",
    )
)

"""intern template
Supports: https://huggingface.co/internlm/internlm-chat-7b
          https://huggingface.co/internlm/internlm-chat-20b
"""
register_conv_template(
    Conversation(
        name="intern",
        system_prompt="",
        messages=[],
        roles=("<|User|>", "<|Bot|>"),
        prompt="<|User|>:{query}<eoh>\n<|Bot|>:",
        sep="<eoa>\n",
        stop_str="<eoa>",
    )
)

"""StarChat template
Supports: https://huggingface.co/HuggingFaceH4/starchat-alpha
          https://huggingface.co/HuggingFaceH4/starchat-beta
"""
register_conv_template(
    Conversation(
        name="starchat",
        system_prompt="<system>\n",
        messages=[],
        roles=("<|user|>", "<|assistant|>"),
        prompt="<|user|>\n{query}<|end|>\n<|assistant|>\n",
        sep="<|end|>\n",
        stop_str="<|end|>",
    )
)

"""llama2 template
Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
"""
register_conv_template(
    Conversation(
        name="llama2",
        system_prompt="<<SYS>>\nYou are a helpful, respectful and honest assistant. "
                      "Always answer as helpfully as possible, while being safe. "
                      "Your answers should not include any harmful, unethical, racist, sexist, "
                      "toxic, dangerous, or illegal content. "
                      "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                      "If a question does not make any sense, or is not factually coherent, "
                      "explain why instead of answering something not correct. "
                      "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST]",
        sep="</s>",
    )
)

"""llama2-zh template
source: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
Supports: https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
"""
register_conv_template(
    Conversation(
        name="llama2-zh",
        system_prompt="[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n [/INST]",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST]",
        sep="</s>",
    )
)

"""mistral template
Supports: https://huggingface.co/mistralai/Mistral-7B-v0.1
          https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
source: https://docs.mistral.ai/llm/mistral-instruct-v0.1
"""
register_conv_template(
    Conversation(
        name="mistral",
        system_prompt="<s>",
        messages=[],
        roles=("[INST]", "[/INST]"),
        prompt="[INST] {query} [/INST]",
        sep="</s>",
    )
)

"""XVERSE template
Supports: https://huggingface.co/xverse/XVERSE-13B-Chat
"""
register_conv_template(
    Conversation(
        name="xverse",
        system_prompt="",
        messages=[],
        roles=("Human", "Assistant"),
        prompt="Human: {query}\n\nAssistant: ",
        sep="</s>",
    )
)

"""Qwen template
Supports: https://huggingface.co/Qwen/Qwen-7B-Chat
chatml: https://xbot123.com/645a461b922f176d7cfdbc2d/
"""
register_conv_template(
    Conversation(
        name="chatml",
        system_prompt="You are a helpful assistant.",
        messages=[],
        roles=("user", "assistant"),
        prompt="<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
        sep="<|im_end|>\n",
        stop_str="<|im_end|>",
    )
)


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name]


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


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

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()

    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Script args: {script_args}")
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    prompt_template = get_conv_template(data_args.template_name)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str  # eos token is required for SFT
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    logger.debug(f"Tokenizer: {tokenizer}")
    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # Get datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        # Loading a dataset from local files.
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {train_data_files}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    max_length = script_args.model_max_length

    def preprocess_function(examples):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        roles = ["human", "gpt"]

        def get_dialog(examples):
            for i, source in enumerate(examples['conversations']):
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role not in roles or data_role != roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]
                if len(source) < 2:
                    continue
                messages = []
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])
                if len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                yield prompt_template.get_dialog(history_messages)

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # eos token
                    target_ids = target_ids[:max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
                labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    def filter_empty_labels(example):
        """Remove empty labels dataset."""
        return not all(label == IGNORE_INDEX for label in example["labels"])

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        with training_args.main_process_first(desc="Train dataset tokenization"):
            train_dataset = train_dataset.shuffle().map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            train_dataset = train_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("Tokenized training example:")
            logger.debug(f"Decode input_ids[0]: {tokenizer.decode(train_dataset[0]['input_ids'])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                               for label in list(train_dataset[0]['labels'])]
            logger.debug(f"Decode labels[0]: {tokenizer.decode(replaced_labels)}")

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset = eval_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    # Load model
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1
        if ddp:
            model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and ZeRO3 are both currently incompatible with QLoRA.")

        config = config_class.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            cache_dir=model_args.cache_dir
        )

        # Set RoPE scaling
        if model_args.rope_scaling is not None:
            if hasattr(config, "use_dynamic_ntk"):  # for Qwen models
                logger.warning("Qwen model does not support RoPE scaling in training.")
            elif hasattr(config, "rope_scaling"):  # for LLaMA and Falcon models
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )
                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and script_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(script_args.model_max_length / current_max_length))
                else:
                    logger.warning(f"The model_max_length({script_args.model_max_length}) is smaller than max "
                                   f"length({current_max_length}). Consider increase model_max_length.")
                    scaling_factor = 1.0

                setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
                logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                    model_args.rope_scaling, scaling_factor
                ))
            else:
                logger.warning("Current model does not support RoPE scaling.")

        # Set FlashAttention-2
        if model_args.flash_attn:
            if getattr(config, "model_type", None) == "llama":
                modeling_llama.LlamaAttention = LlamaFlashAttention2
                modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
                logger.info("Using FlashAttention-2 for faster training and inference.")
            elif getattr(config, "model_type", None) == "qwen":
                logger.info("Qwen models automatically enable FlashAttention if installed.")
            else:
                logger.warning("Current model does not support FlashAttention-2.")
        elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
            modeling_llama.LlamaAttention = LlamaShiftShortAttention
            logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
                           " is RTX4090, A100 or H100.")

        # Set shift short attention (S^2-Attn)
        if model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                logger.info("Using shift short attention with group_size_ratio=1/4.")
            else:
                logger.warning("Current model does not support shift short attention.")

        load_in_4bit = model_args.load_in_4bit
        load_in_8bit = model_args.load_in_8bit
        load_in_8bit_skip_modules = None
        if load_in_8bit or load_in_4bit:
            logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if script_args.modules_to_save is not None:
                load_in_8bit_skip_modules = script_args.modules_to_save.split(',')
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map=model_args.device_map,
            trust_remote_code=model_args.trust_remote_code,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                load_in_8bit_skip_modules=load_in_8bit_skip_modules,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            ) if script_args.qlora else None,
        )

        # Fix ChatGLM2 and ChatGLM3 LM head
        if getattr(config, "model_type", None) == "chatglm":
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

        # Set NEFTune trick for fine-tuning
        if model_args.neft_alpha > 0:
            input_embed = model.get_input_embeddings()
            if isinstance(input_embed, torch.nn.Embedding):
                def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                    embeddings = input_embed.__class__.forward(self, x)
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = model_args.neft_alpha / (dims ** 0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                    return embeddings

                input_embed.forward = MethodType(noisy_forward, input_embed)
                logger.info("Using noisy embedding with alpha={:.2f}".format(model_args.neft_alpha))
            else:
                logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")
    else:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    if script_args.use_peft:
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
        if script_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        logger.info("Fine-tuning method: Full parameters training")
        # model = model.float()
        print_trainable_parameters(model)

    # Initialize our Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,  # for shift short attention
    )
    # Initialize our Trainer
    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        if trainer.is_world_process_zero():
            sample = next(iter(trainer.get_train_dataloader()))
            logger.debug(f"Train dataloader example: {sample}")
            logger.debug(f"Detail input_ids: {list(sample['input_ids'])[:3]}, \nlabels: {list(sample['labels'])[:3]}")
            logger.debug(f"Decode input_ids[0]: {tokenizer.decode(sample['input_ids'][0])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id for label in
                               sample['labels'][0]]
            logger.debug(f"Decode labels[0]: {tokenizer.decode(replaced_labels)}")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        model.config.use_cache = True  # enable cache after training
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"

        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()