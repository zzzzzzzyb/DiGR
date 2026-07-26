import torch
import transformers
import argparse
import os
import math

os.environ["WANDB_MODE"] = 'offline'
import json

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, \
    PreTrainedModel, GPT2LMHeadModel, Qwen3ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from utils import *
from collator import Collator

class PreExpLM(PreTrainedModel):
    # 新版 Transformers gradient checkpointing：不要实现 _set_gradient_checkpointing
    supports_gradient_checkpointing = True

    def __init__(self, model, config, num_heads=4, item_dict=None, tokenizer=None):
        super().__init__(config)
        self.peft_or_base = model
        base = model.get_base_model() if hasattr(model, "get_base_model") else model

        # Qwen/LLaMA 类一般是 CausalLM: base.model 是 decoder，base.lm_head 是 head
        self.model = getattr(base, "model", base)
        self.lm_head = getattr(base, "lm_head", None)
        if self.lm_head is None:
            raise ValueError("无法找到 lm_head，请确认 base_model 是 CausalLM 模型。")
        self.lm_head = model.lm_head
        self.tokenizer = tokenizer
        self.item_dict = {int(k): tokenizer.convert_tokens_to_ids(v) for k, v in item_dict.items()}
        self.reverse_dict = {"".join(v): int(k) for k, v in item_dict.items()}

        self.model_parallel = False
        self.device_map = None
        self.is_parallelizable = True
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # 初始温度≈0.07

        self.item_token_indices = torch.tensor(list(self.item_dict.values()))

        # Trainer 会调用 gradient_checkpointing_enable/disable；这里保存 wrapper 自身状态
        self.gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self, "gradient_checkpointing", False)) or bool(
            getattr(self.model, "gradient_checkpointing", False)
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # gradient_checkpointing_kwargs 在新版 Transformers 可能会被忽略；这里只做兼容
        self.gradient_checkpointing = True

        # 只对最外层模型（通常是 PEFT wrapper）调用一次，避免重复启用导致数值不稳定
        target = self.peft_or_base
        if not hasattr(target, "gradient_checkpointing_enable") and hasattr(self.model, "gradient_checkpointing_enable"):
            target = self.model

        if hasattr(target, "gradient_checkpointing_enable"):
            try:
                target.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            except TypeError:
                target.gradient_checkpointing_enable()
        elif hasattr(target, "gradient_checkpointing"):
            try:
                target.gradient_checkpointing = True
            except Exception:
                pass

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

        target = self.peft_or_base
        if not hasattr(target, "gradient_checkpointing_disable") and hasattr(self.model, "gradient_checkpointing_disable"):
            target = self.model

        if hasattr(target, "gradient_checkpointing_disable"):
            try:
                target.gradient_checkpointing_disable()
            except Exception:
                pass
        elif hasattr(target, "gradient_checkpointing"):
            try:
                target.gradient_checkpointing = False
            except Exception:
                pass

    def _assert_finite(self, name: str, x: torch.Tensor):
        if x is None:
            return
        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).nonzero(as_tuple=False)
            raise FloatingPointError(
                f"[NaN/Inf detected] {name}: dtype={x.dtype}, shape={tuple(x.shape)}, "
                f"first_bad_index={bad[0].tolist() if bad.numel() else None}, "
                f"min={x.nan_to_num().min().item()}, max={x.nan_to_num().max().item()}"
            )
        
    def forward(
            self,
            input_ids = None,
            past_key_values = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,
            encoder_hidden_states = None,
            encoder_attention_mask = None,
            labels = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            **kwargs,
    ):  

        # 开启 gradient checkpointing 时必须禁用 KV cache，否则可能报错/无效
        if self.is_gradient_checkpointing and (use_cache is None or use_cache is True):
            use_cache = False
        

        
        item_embeddings = self.model.embed_tokens(self.item_token_indices.to(input_ids.device))  # (num_items, 4, hidden_size)
        
        self._assert_finite("embed_tokens.weight", self.model.embed_tokens.weight)
        self._assert_finite("item_embeddings(before_gate)", item_embeddings)

        # gate_weights = torch.reshape(item_embeddings, (item_embeddings.shape[0], -1))  # (num_items, 4*hidden_size)
        # gate_weights = self.gate_mlp(gate_weights)  # (num_items, 4)
        # gate_weights = torch.softmax(gate_weights, dim=-1)  # (num_items, 4)
        # item_embeddings = torch.sum(item_embeddings * gate_weights.unsqueeze(-1), dim=1)  # (num_items, hidden_size)
        item_embeddings = item_embeddings.mean(dim=1)
        # self._assert_finite("item_embeddings(after_gate)", item_embeddings)

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        hidden_states = hidden_states[:, -1, :]

        self._assert_finite("hidden_states(last_token)", hidden_states)

        if self.model_parallel:
            torch.cuda.set_device(self.model.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        hs = hidden_states.float()
        ie = item_embeddings.float()
        hs = F.normalize(hs, dim=-1)
        ie = F.normalize(ie, dim=-1)

        logit_scale = self.logit_scale.clamp(min=math.log(1/100.0), max=math.log(100.0)).exp()
        logits = logit_scale * (hs @ ie.t())
        self._assert_finite("logits", logits)
        # logits = hidden_states @ item_embeddings.t()
        loss = None 
        
        if labels is not None:
            labels = [label[label != -100][:-1] for label in labels] 
            labels = [self.tokenizer.convert_ids_to_tokens(label) for label in labels]
            try:
                labels = [self.reverse_dict["".join(label)] for label in labels]
            except Exception:
                idx = labels.index([])
                print("Error label:", idx)
                labels.pop(idx)
                logits = torch.cat([logits[:idx], logits[idx+1:]], dim=0)
                labels = [self.reverse_dict["".join(label)] for label in labels]
            loss = self.loss_fct(logits, torch.tensor(labels).to(logits.device))
        
        
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": transformer_outputs.past_key_values,
            "hidden_states": transformer_outputs.hidden_states,
            "attentions": transformer_outputs.attentions,
        }
    
    

def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = 'auto'
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}


    config = AutoConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length = args.model_max_length,
        padding_side="right",
    )
    tokenizer.pad_token_id = 0
    gradient_checkpointing = False

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    collator = Collator(args, tokenizer)
    pre_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )
    with open(f'./data/{args.dataset}/{args.dataset}.index.json', 'r') as f:
        items = json.load(f)
    # reverse_dict = {"".join(v) : int(k) for k,v in items.items()}

    pre_model.resize_token_embeddings(len(tokenizer))

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        modules_to_save=args.lora_modules_to_save.split(",")
    )
    pre_model = get_peft_model(pre_model, lora_config)
    pre_model.get_input_embeddings().weight.requires_grad_(True)
    if local_rank == 0:
        pre_model.print_trainable_parameters()

    model = PreExpLM(pre_model, config, num_heads=4, item_dict=items, tokenizer=tokenizer)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=True,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=None,
            eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
            remove_unused_columns=False,
            # PEFT wrappers may hide the base model's forward signature from Trainer.
            # Declare the label field explicitly so evaluation aggregates the returned loss.
            label_names=["labels"],
            save_safetensors=False,
            max_grad_norm=1.0,
            # max_steps=1500,
        ),
        processing_class=tokenizer,
        data_collator=collator,
    )
    model.config.use_cache = False

    trainer.train(
        resume_from_checkpoint=False
    )

    # === 统一保存：只保存 LoRA adapter + Trainer 状态，不处理 gate_mlp ===
    pre_model.save_pretrained(args.output_dir)
    trainer.save_state()
    with open(os.path.join(args.output_dir, 'logit_scale.txt'), 'w') as f:
        f.write(model.logit_scale)
    print(model.logit_scale)

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
