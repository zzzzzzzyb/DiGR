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
import torch.nn.functional as F


def test(args):

    print(args)
    #get real items
    with open(f'./data/{args.dataset}/{args.dataset}.index.json', 'r') as f:
        items = json.load(f)
    with open('./data/Instruments/Instruments.inter.json', 'r') as f:
        inter = json.load(f)
    reverse_dict = {"".join(v) : int(k) for k,v in items.items()}
    items = list(items.values())
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, fix_mistral_regex=True)
    items = [tokenizer.convert_tokens_to_ids(item) for item in items]
    
    
    device_map = {"": args.gpu_id}
    device = torch.device("cuda", args.gpu_id)
    config = AutoConfig.from_pretrained(args.ckpt_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation='flash_attention_2'
    )
    base_model.resize_token_embeddings(len(tokenizer))
    lora_model = PeftModel.from_pretrained(base_model, args.ckpt_path, device_map=device_map)
    lora_model.eval()
    # state_dict = torch.load(f"{args.ckpt_path}/pytorch_model.bin")
    with open(f'./data/{args.dataset}/{args.dataset}.index.json', 'r') as f:
        item_dict = json.load(f)
    custom_model = PreExpLM(model=lora_model, config=config, num_heads=4 ,item_dict=item_dict, tokenizer=tokenizer)
    # custom_model.load_state_dict(state_dict, strict=False)
    custom_model.to(device)
    custom_model.eval()

    extra_path = os.path.join(args.ckpt_path, "extra_modules.pt")
    if os.path.exists(extra_path):
        extra = torch.load(extra_path, map_location="cpu")
        if "gate_mlp" in extra:
            custom_model.gate_mlp.load_state_dict(extra["gate_mlp"], strict=True)

    item_table = []
    for item in items:
        item = torch.tensor(item).to(device)
        item_emb = torch.mean(custom_model.model.embed_tokens(item), dim=0).detach().cpu().numpy()
        # item_emb = custom_model.model.embed_tokens(item).detach().cpu().numpy()
        item_table.append(item_emb)
        
    item_table = torch.tensor(item_table).to(device)
    item_table = F.normalize(item_table)
    # item_table = custom_model.item_mapper(torch.tensor(item_table).to(device)).detach().cpu().numpy()
    # item_table = torch.tensor(item_table).to(device)
    # gate_weights = torch.reshape(item_table, (item_table.shape[0], -1))  # (num_items, 4*hidden_size)
    # gate_weights = custom_model.gate_mlp(gate_weights)  # (num_items, 4)
    # gate_weights = torch.softmax(gate_weights, dim=-1)  # (num_items, 4)
    # item_table = torch.sum(item_table * gate_weights.unsqueeze(-1), dim=1)  # (num_items, hidden_size)

    # B, N, D = item_table.shape
    # q = custom_model.W_q(custom_model.query).expand(B, 1, D)
    # k = custom_model.W_k(item_table)
    # v = custom_model.W_v(item_table)
    # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    # attn = torch.softmax(scores, dim=-1)
    # item_table = torch.matmul(attn, v).squeeze(1)  # (B, D)
#     item_table = item_table.detach().cpu().numpy()

#     item_table = np.array(item_table).astype('float32')
#     index = faiss.IndexFlatL2(item_table.shape[1])
#     index.add(item_table)

    custom_model.eval()
    
    custom_model.to(device)

    
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

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)
    

    metrics = args.metrics.split(",")
    all_prompt_results = []

    print("data num:", len(test_data))


    with torch.no_grad():
        for prompt_id in prompt_ids:
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0
            # os.system(f'touch ./results/{args.dataset}_gpt2_multi/{prompt_id}.txt')
            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                total += len(targets)
                target_item_index = [reverse_dict[targets[0]]]
                # generate outputs
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                output_hidden_states = custom_model.model(input_ids=input_ids,
                                                                attention_mask=attention_mask,)
                hidden_states = output_hidden_states[0][:, -1, :]
                
                hidden_states = F.normalize(hidden_states)

                # output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # useful
                # hidden_states = custom_model.hidden_mapper(hidden_states)
                # hidden_states = hidden_states.cpu().numpy()
                # distances, indices = index.search(hidden_states, k=10)

                # topk_res, sub_res = get_topk_results(output, scores, targets, args.num_beams,
                #                                      all_items=all_items if args.filter_items else None)

                # useful
                # indices = indices[0].tolist()
                predeict_scores = torch.matmul(hidden_states, item_table.transpose(0,1))
                _, indices = torch.topk(predeict_scores, k=20)
                indices = indices[0].tolist()
                # print(indices)
                with open('./res_for_diversity.txt', 'a') as f:
                    # print(indices)
                    # print(','.join(indices))
                    f.write(','.join(map(str, indices)))
                    f.write('\n')
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

                if (step + 1) % 100 == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
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

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test(args)

