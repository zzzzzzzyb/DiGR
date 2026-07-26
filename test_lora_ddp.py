from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from train_lrc import PreExpLM
import torch
from safetensors.torch import load_file
import argparse
from transformers import GPT2LMHeadModel
from utils import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import faiss
import numpy as np
import math
from peft import PeftModel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def test(args):

    # ---- multi-GPU inference (DDP via torchrun) ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if ddp else 0
    if rank == 0:
        print(args)
    #get real items
    with open(f'./data/{args.dataset}/{args.dataset}.index.json', 'r') as f:
        items = json.load(f)
    with open('./data/Instruments/Instruments.inter.json', 'r') as f:
        inter = json.load(f)
    reverse_dict = {"".join(v) : int(k) for k,v in items.items()}
    items = list(items.values())
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    items = [tokenizer.convert_tokens_to_ids(item) for item in items]
    
    
    sharded = (not ddp) and (int(getattr(args, "gpu_id", 0)) == -1)

    if ddp:
        device_map = {"": local_rank}
        device = torch.device("cuda", local_rank)
    elif sharded:
        device_map = "auto"
        device = None
    else:
        device_map = {"": args.gpu_id}
        device = torch.device("cuda", args.gpu_id)
    config = AutoConfig.from_pretrained(args.ckpt_path)

    torch_dtype = None
    if hasattr(args, "bf16") and args.bf16:
        torch_dtype = torch.bfloat16
    elif hasattr(args, "fp16") and args.fp16:
        torch_dtype = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map
    )
    base_model.resize_token_embeddings(len(tokenizer))
    lora_model = PeftModel.from_pretrained(base_model, args.ckpt_path, device_map=device_map)
    lora_model.eval()
    # state_dict = torch.load(f"{args.ckpt_path}/pytorch_model.bin", map_location="cpu")
    with open(f'./data/{args.dataset}/{args.dataset}.index.json', 'r') as f:
        item_dict = json.load(f)
    custom_model = PreExpLM(model=lora_model, config=config, num_heads=4 ,item_dict=item_dict, tokenizer=tokenizer)
    # print(custom_model.model)
    # custom_model.load_state_dict(state_dict, strict=False)
    custom_model.eval()

    def _get_backbone(m):
        # 兼容旧实现：有的 wrapper 暴露 transformer，有的暴露 model
        if hasattr(m, "transformer"):
            return m.transformer
        if hasattr(m, "model"):
            return m.model
        return m

    backbone = _get_backbone(custom_model)

    def _get_input_device(m):
        try:
            emb = m.get_input_embeddings()
            if emb is not None:
                return emb.weight.device
        except Exception:
            pass
        # fallback: 尝试常见字段
        if hasattr(backbone, "embed_tokens"):
            return backbone.embed_tokens.weight.device
        if hasattr(backbone, "wte"):
            return backbone.wte.weight.device
        # 最后兜底：非分片时用显式 device
        return device if device is not None else torch.device("cuda", 0)

    input_device = _get_input_device(custom_model)

    extra_path = os.path.join(args.ckpt_path, "extra_modules.pt")
    if os.path.exists(extra_path):
        extra = torch.load(extra_path, map_location="cpu")
        if "gate_mlp" in extra:
            custom_model.gate_mlp.load_state_dict(extra["gate_mlp"], strict=True)

    # ---- build item embedding table (GPU-friendly, model-agnostic) ----
    input_emb = None
    try:
        input_emb = custom_model.get_input_embeddings()
    except Exception:
        input_emb = None
    if input_emb is None:
        if hasattr(backbone, "embed_tokens"):
            input_emb = backbone.embed_tokens
        elif hasattr(backbone, "wte"):
            input_emb = backbone.wte
    
    
    input_emb = custom_model.model.embed_tokens
    if input_emb is None:
        raise ValueError("Cannot locate input embeddings for building item_table.")

    item_table = []
    for item in items:
        item_ids = torch.tensor(item, device=input_device)
        emb = input_emb(item_ids)
        # item token数一般是固定长度（如4）；取均值更稳健
        emb = emb.mean(dim=0)
        item_table.append(emb.detach().float().cpu().numpy())

    item_table = np.array(item_table).astype('float32')

    item_table = np.array(item_table).astype('float32')
    index = faiss.IndexFlatL2(item_table.shape[1])
    index.add(item_table)

    custom_model.eval()

    
    if args.test_prompt_ids == "all":
        if args.test_task.lower() == "seqrec":
            prompt_ids = range(len(all_prompt["seqrec"]))
        elif args.test_task.lower() == "itemsearch":
            prompt_ids = range(len(all_prompt["itemsearch"]))
        elif args.test_task.lower() == "fusionseqrec":
            prompt_ids = range(len(all_prompt["fusionseqrec"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    sampler = None
    if ddp:
        sampler = DistributedSampler(test_data, shuffle=True, drop_last=False)

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    metrics = args.metrics.split(",")
    all_prompt_results = []

    if rank == 0:
        print("data num:", len(test_data))


    with torch.no_grad():
        for prompt_id in prompt_ids:
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0
            # os.system(f'touch ./results/{args.dataset}_gpt2_multi/{prompt_id}.txt')
            if ddp and sampler is not None:
                sampler.set_epoch(prompt_id)

            pbar = tqdm(test_loader, disable=(rank != 0))
            for step, batch in enumerate(pbar):
                inputs = batch[0]
                if device is not None:
                    inputs = inputs.to(device)
                else:
                    # 分片模型：把输入放到 embedding 所在设备即可，HF 会自动跨设备跑完 forward
                    inputs = {k: v.to(input_device) for k, v in inputs.items()}
                targets = batch[1]
                total += len(targets)
                target_item_index = [reverse_dict[targets[0]]]
                # generate outputs
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                output_hidden_states = custom_model.model(input_ids=input_ids,
                                                attention_mask=attention_mask,)
                hidden_states = output_hidden_states[0][:, -1, :]

                # output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # useful
                # hidden_states = custom_model.hidden_mapper(hidden_states)
                hidden_states = hidden_states.cpu().numpy()
                distances, indices = index.search(hidden_states, k=10)

                # topk_res, sub_res = get_topk_results(output, scores, targets, args.num_beams,
                #                                      all_items=all_items if args.filter_items else None)

                # useful
                # indices = indices[0].tolist()
                # predeict_scores = torch.matmul(hidden_states, item_table.transpose(0,1))
                # _, indices = torch.topk(predeict_scores, k=10)
                indices = indices[0].tolist()
                topk_res = [[0] * 10]
                for i in range(10):
                    if indices[i] == int(target_item_index[0]):
                        topk_res[0][i] = 1
                        break
                
                # item_rank = custom_model.mapper(hidden_states)[0]
                # scores = F.softmax(item_rank, dim=-1)
                # top10 = torch.topk(scores, k=10).indices.tolist()
                # topk_res = [[0] * 10]
                # for i in range(10):
                #     if top10[i] == int(target_item_index[0]):
                #         topk_res[0][i] = 1
                #         break

                # print(target_item_index, indices)
                # with open(f'./results/{args.dataset}_gpt2_multi/{prompt_id}_top10.txt', 'a') as f:
                #     for sorted_pred in sub_res:
                #         f.write(sorted_pred[0])
                #         # print(sorted_pred[0])
                #         f.write('\n')
                # with open(f'./results/{args.dataset}_gpt2_multi/gt.txt', 'a') as f:
                #     f.write(targets[0])
                #     f.write('\n')
                batch_metrics_res = get_metrics_results(topk_res, metrics)
                # print(batch_metrics_res)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if (step + 1) % 10 == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    if rank == 0:
                        print(temp)

            # ---- aggregate across ranks ----
            if ddp:
                total_t = torch.tensor([total], device=device, dtype=torch.long)
                dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
                total_all = int(total_t.item())

                for m in metrics_results:
                    v = torch.tensor([metrics_results[m]], device=device, dtype=torch.float64)
                    dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    metrics_results[m] = (v.item() / max(total_all, 1))
                total = total_all
            else:
                for m in metrics_results:
                    metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
            if rank == 0:
                print("======================================================")
                print("Prompt {} results: ".format(prompt_id), metrics_results)
                print("======================================================")
                print("")

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res) / len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    if rank == 0:
        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")

    save_data = {}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    if rank == 0:
        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)

    if ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test(args)

