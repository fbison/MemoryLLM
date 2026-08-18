import sklearn
import pandas as pd
import argparse, os, sys, glob, datetime, yaml
import torch
import time
import sys
import numpy as np
from tqdm import trange
from omegaconf import OmegaConf
from collections import OrderedDict
import copy
from transformers import AutoTokenizer
from modeling_memoryllm import MemoryLLM
from modeling_mplus import MPlus
from torch.utils.data import Dataset, DataLoader
from dataset.nq import NQDataset
from dataset.squad import SQuADDataset
import json
import pandas as pd
from tqdm import tqdm
from functools import partial



def collate_fn_qa(data, tokenizer, max_length, num_tokens, 
                eval_max_length=None,
                add_special_tokens=False, 
                end_special_token=None,
                mask_strategy=None,
                mask_ratio=None,
                padding='longest'):
                
    eval_max_length = max_length if eval_max_length is None else eval_max_length

    contexts, questions, answers, unrelated_contexts = zip(*data)

    if end_special_token is not None:
        answers = [x + end_special_token for x in list(answers)]
    
    contexts_tokenized = tokenizer(list(contexts), 
                                   max_length=max_length, 
                                   padding=padding,
                                   truncation=True, 
                                   return_tensors='pt',
                                   add_special_tokens=add_special_tokens)
    questions_tokenized = tokenizer(list(questions), 
                                    max_length=eval_max_length,
                                    truncation=True, 
                                    padding='longest',
                                    return_tensors='pt',
                                    add_special_tokens=add_special_tokens)
    answers_tokenized = tokenizer(list(answers),
                                    max_length=eval_max_length,
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt',
                                    add_special_tokens=add_special_tokens)

    # eg: batch_size: 4
    # eg: time_steps: 8
    # then unrelated_contexts would be 4 * 8; 
    # I need it to be 8 * 4

    unrelated_contexts = np.array(unrelated_contexts).transpose().tolist()

    all_unrelated_contexts = {}
    all_unrelated_contexts_mask = {}
    for i in range(len(unrelated_contexts)):
        all_unrelated_contexts[i] = tokenizer(unrelated_contexts[i],
                                    max_length=max_length,
                                    # padding='max_length',
                                    truncation=True,
                                    padding=padding,
                                    return_tensors='pt',
                                    add_special_tokens=add_special_tokens)
        all_unrelated_contexts_mask[i] = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(all_unrelated_contexts[i].input_ids.shape[0], 1),
                            all_unrelated_contexts[i].attention_mask], dim=-1)

    # Create attention masks with total_length
    contexts_attention_mask = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1), 
                                         contexts_tokenized.attention_mask], dim=-1)
    questions_attention_mask = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1), 
                                          questions_tokenized.attention_mask], dim=-1)
    answers_attention_mask = torch.cat([torch.tensor([1]*num_tokens).unsqueeze(0).repeat(contexts_tokenized.input_ids.shape[0], 1),
                                         answers_tokenized.attention_mask], dim=-1)

    outputs = (contexts_tokenized.input_ids, contexts_attention_mask, \
        questions_tokenized.input_ids, questions_attention_mask, \
        answers_tokenized.input_ids, answers_attention_mask)
    
    for i in range(len(all_unrelated_contexts)):
        outputs += (all_unrelated_contexts[i].input_ids,)
        outputs += (all_unrelated_contexts_mask[i],)
        
    return outputs



# # Configure logging
# logging.basicConfig(
#     level=print,  # Set the desired log level (e.g., INFO, DEBUG, WARNING)
#     filename='/fsx-Training/shopqa-training-fsx-prod-us-east-1/wangyuu/log_qa.txt',  # Specify the file to write the log output
#     filemode='a',       # 'a' stands for "append", which appends log output to the file
#     format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log message format
# )

# # Test the logging
# print('This is an informational message.')
# logging.warning('This is a warning message.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--model",
        type=str,
        nargs="?",
        default=None,
        help="model path",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=1
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=['naturalqa']
    )
    parser.add_argument(
        "--nuc",
        type=int,
        default=1
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=200
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        '--backup_memory',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--put_memory_on_cpu',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--related_position',
        default='begin',
        choices=['begin', 'end', 'random']
    )
    parser.add_argument(
        "--num_tokens",
        default=256,
        type=int
    )
    parser.add_argument(
        "--split_model",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="Evaluation step interval for generation (default: 1, evaluate all steps. For 10k token increments up to 160k with nuc=320, use --eval_interval 20)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker count (default: 0 to prevent OS open file descriptor exhaustion under large distractor counts)"
    )

    return parser


def load_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.read().splitlines()
    data = [json.loads(line) for line in lines]
    return pd.DataFrame(data)

def run_qa(model, tokenizer, dataset, step=1, output_filename=None):

    if dataset == 'naturalqa':
        dataset = NQDataset(
            filename = "./data/nq/v1.0-simplified_nq-dev-all.jsonl",
            num = opt.num_samples,
            num_unrelated_contexts=step,
            tokenizer=tokenizer,
            tokenizer_path=None
        )
        
    elif dataset == 'squad':

        dataset = SQuADDataset(
            filename = './data/squad/dev-v2.0.json',
            num = opt.num_samples,
            num_unrelated_contexts=step,
            tokenizer=tokenizer,
            tokenizer_path=None
        )

    collate_fn_with_params = partial(collate_fn_qa, 
            tokenizer=tokenizer, 
            max_length=512,
            num_tokens=opt.num_tokens,
            add_special_tokens=False,
            end_special_token="</s>",)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn_with_params)
    print(f"Loaded {len(dataset)} samples for development")

    model.eval()

    preds_with_context = []
    contexts_middle = {
        f"step_{idx}": []
        for idx in range(step+1)
    }
    middle_outputs = {
        f"step_{idx}": []
        for idx in range(step+1)
    }
    contexts = []
    questions = []
    targets = []

    if opt.backup_memory:
        backup_memory = model.memory.data.detach().cpu().clone()
    else:
        backup_memory = None

    with torch.no_grad():

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            
            batch = [x.cuda() for x in batch]

            context_ids, context_attention_mask, \
            sentence_ids, sentence_attention_mask, \
            answer_ids, answer_attention_mask = batch[:6]

            unrelated_contexts_and_mask = batch[6:]

            unrelated_contexts_ids = []
            unrelated_contexts_attention_masks = []
            for i in range(len(unrelated_contexts_and_mask) // 2):
                unrelated_contexts_ids.append(unrelated_contexts_and_mask[i*2])
                unrelated_contexts_attention_masks.append(unrelated_contexts_and_mask[i*2 + 1])
            
            if opt.related_position == 'begin':
                contexts_ids = [context_ids] + unrelated_contexts_ids
                contexts_attention_masks = [context_attention_mask] + unrelated_contexts_attention_masks
            
            elif opt.related_position == 'end':
                if batch_idx == 0:
                    contexts_ids = unrelated_contexts_ids + [context_ids]
                    contexts_attention_masks = unrelated_contexts_attention_masks + [context_attention_mask]
                else:
                    contexts_ids = [context_ids]
                    contexts_attention_masks = [context_attention_mask]
            
            else: 
                # randomly split unrelated_contexts_ids into two parts:
                split_index = np.random.randint(0, len(unrelated_contexts_ids)+1)
                contexts_ids = unrelated_contexts_ids[:split_index] + [context_ids] + unrelated_contexts_ids[split_index:]
                contexts_attention_masks = unrelated_contexts_attention_masks[:split_index] + [context_attention_mask] + unrelated_contexts_attention_masks[split_index:]
            
            if backup_memory is not None:
                model.memory.data = backup_memory.clone().to(model.memory.device)
            
            sentence_attention_mask = torch.cat([
                        torch.ones(sentence_attention_mask.shape[0], model.num_tokens*(model.num_blocks-1)).cuda(),
                        sentence_attention_mask,
                    ], dim=1)

            for idx, (ids, mask)in enumerate(zip(contexts_ids, contexts_attention_masks)):

                model.inject_memory(
                    ids,
                    mask,
                    update_memory=True
                )

                if opt.related_position == 'begin':
                    if idx == 0 or idx % opt.eval_interval == 0:
                        output = model.generate(
                            inputs=sentence_ids, 
                            attention_mask=sentence_attention_mask,
                            max_new_tokens=10,
                            pad_token_id=tokenizer.pad_token_id
                        )[:, len(sentence_ids[0]):][0].detach().cpu()

                        middle_outputs[f"step_{idx}"].append(output)
                
                contexts_middle[f"step_{idx}"].append(ids[0].detach().cpu())

            if opt.related_position == 'end' and batch_idx == 0:
                backup_memory = model.memory.data.detach().cpu().clone()

            if opt.related_position != 'begin':

                output = model.generate(
                    inputs=sentence_ids, 
                    attention_mask=sentence_attention_mask,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.pad_token_id
                )[:, len(sentence_ids[0]):][0].detach().cpu()

                middle_outputs[f"step_0"].append(output)

            questions.append(sentence_ids.detach().cpu().numpy()[0])
            targets.append(answer_ids.detach().cpu().numpy()[0])

            if output_filename is not None:
                dec_targets = tokenizer.batch_decode(targets)
                dec_middle_outputs = {
                    key: tokenizer.batch_decode(value)
                    for key, value in middle_outputs.items()
                }
                dec_questions = tokenizer.batch_decode(questions) if len(questions) > 0 else []
                dec_contexts_middle = {
                    key: tokenizer.batch_decode(value)
                    for key, value in contexts_middle.items()
                }
                save_formatted_json(output_filename, opt, dec_targets, dec_middle_outputs, dec_contexts_middle, dec_questions)

    targets = tokenizer.batch_decode(targets)
    middle_outputs = {
        key: tokenizer.batch_decode(value)
        for key,value in middle_outputs.items()
    }

    if len(questions) > 0:
        questions = tokenizer.batch_decode(questions)
    
    contexts_middle = {
        key: tokenizer.batch_decode(value)
        for key,value in contexts_middle.items()
    }

    return middle_outputs, targets, contexts_middle, questions

def calculate_exact_hit_accuracy(predictions, targets):
    count = 0
    hit = 0
    for i in range(len(predictions)):
        if targets[i].replace("</s>", "").strip() in predictions[i]:
            hit += 1
        count += 1
    return hit / count if count > 0 else 0.0

def save_formatted_json(filename, opt, targets, middle_outputs, contexts_middle, questions):
    metrics = {}
    if len(targets) > 0:
        if 'step_0' in middle_outputs and len(middle_outputs['step_0']) == len(targets):
            metrics['step_0'] = round(calculate_exact_hit_accuracy(middle_outputs['step_0'], targets), 4)
        if opt.related_position == 'begin':
            for step_key, outputs_list in middle_outputs.items():
                if step_key == 'step_0': continue
                if len(outputs_list) == len(targets) and len(targets) > 0:
                    metrics[step_key] = round(calculate_exact_hit_accuracy(outputs_list, targets), 4)

    generated_results = {
        'param': {
            'model': opt.model,
            'max_steps': opt.max_steps,
            'num_unrelated_contexts': opt.nuc,
            'eval_interval': getattr(opt, 'eval_interval', 1),
            'test_samples': len(targets)
        },
        'metrics': metrics
    }

    for i in range(len(targets)):
        generated_results[str(i)] = {
            'w/ context': middle_outputs['step_0'][i],
            'target': targets[i],
            'context': contexts_middle['step_0'][i] if len(contexts_middle['step_0']) > 0 else None,
            'question': questions[i] if len(questions) > 0 else None,
        }
        
        if opt.related_position != 'end':
            for key in contexts_middle.keys():
                if key == 'step_0': continue
                generated_results[str(i)].update({
                    f"contexts_{key}": contexts_middle[key][i]
                })

        if opt.related_position == 'begin':
            for key in middle_outputs.keys():
                if key == 'step_0': continue
                if len(middle_outputs[key]) > i:
                    generated_results[str(i)].update({
                        f"prediction_{key}": middle_outputs[key][i]
                    })

    with open(filename, "w") as file:
        json.dump(generated_results, file, indent=4)

def load_model_and_tokenizer(model_path, split_model=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if "mplus" in model_path.lower():
        print(f"Loading MPlus model from {model_path}...")
        try:
            if not split_model:
                model = MPlus.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).cuda()
            else:
                model = MPlus.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='auto')
        except Exception as e:
            print(f"Failed loading MPlus with flash_attention_2: {e}. Trying default attention implementation...")
            if not split_model:
                model = MPlus.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
            else:
                model = MPlus.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
        model = model.to(torch.bfloat16)
        model.put_ltm_to_numpy()
    else:
        print(f"Loading MemoryLLM model from {model_path}...")
        if not split_model:
            model = MemoryLLM.from_pretrained(model_path).cuda()
        else:
            model = MemoryLLM.from_pretrained(model_path, device_map='auto')
        model = model.to(torch.float16)
        
    return model, tokenizer

if __name__ == "__main__":

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    print(opt)

    model = None
    tokenizer = None

    if opt.put_memory_on_cpu:
        model.convert_memory_to_cpu()

    original_nuc = opt.nuc

    for dataset in opt.datasets:

        if opt.related_position == 'end' or opt.related_position == 'random' and original_nuc == -1:
            accs = []
            for nuc in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:

                opt.nuc = nuc

                print(f"Running {dataset} dataset with nuc {opt.nuc}")
                # filename = f"./results/results_{dataset}_{os.path.basename(opt.model)}_nuc_{opt.nuc}{'_backup' if opt.backup_memory else ''}.json"
                if not os.path.exists(f"results/{dataset}"):
                    os.makedirs(f"results/{dataset}", exist_ok=True)
                if not os.path.exists(f"results/{dataset}/{os.path.basename(opt.model)}"):
                    os.makedirs(f"results/{dataset}/{os.path.basename(opt.model)}", exist_ok=True)
                filename = f"results/{dataset}/{os.path.basename(opt.model)}/results_samples_{opt.num_samples}_nuc_{opt.nuc}{'_backup' if opt.backup_memory else ''}_{opt.related_position}.json"

                is_cached = False
                if os.path.exists(filename):
                    try:
                        cached_data = json.load(open(filename, 'r'))
                        if cached_data.get('param', {}).get('test_samples', 0) >= opt.num_samples:
                            generated_results = cached_data
                            is_cached = True
                        else:
                            print(f"Cached file {filename} has {cached_data.get('param', {}).get('test_samples', 0)} samples, which is fewer than requested {opt.num_samples}. Re-running evaluation...")
                    except Exception:
                        pass

                if not is_cached:
                    if model is None or tokenizer is None:
                        model, tokenizer = load_model_and_tokenizer(opt.model, opt.split_model)

                    middle_outputs, targets, contexts_middle, questions = run_qa(model, tokenizer, dataset, step=opt.nuc, output_filename=filename)

                    generated_results = {
                        'param': {
                            'model': opt.model,
                            'max_steps': opt.max_steps,
                            'num_unrelated_contexts': opt.nuc,
                            'test_samples': len(targets)
                        }
                    }

                    for i in range(len(targets)):
                        
                        generated_results[str(i)] = {
                            'w/ context': middle_outputs['step_0'][i],
                            'target': targets[i],
                            'context': contexts_middle['step_0'][i] if len(contexts_middle['step_0']) > 0 else None,
                            'question': questions[i] if len(questions) > 0 else None,
                        }
                        
                        if opt.related_position != 'end':
                            for key in contexts_middle.keys():
                                if key == 'step_0': continue
                                generated_results[str(i)].update({
                                    f"contexts_{key}": contexts_middle[key][i]
                                })

                        if opt.related_position == 'begin':

                            for key in middle_outputs.keys():
                                
                                if key == 'step_0': continue

                                generated_results[str(i)].update({
                                    f"prediction_{key}": middle_outputs[key][i]
                                })

                    with open(filename, "w") as file:
                        json.dump(generated_results, file, indent=4)
                    file.close()
                
                print("Step 0:")
                acc = calculate_exact_hit_accuracy([x['w/ context'] for x in list(generated_results.values())[1:]],
                                                [x['target'] for x in list(generated_results.values())[1:]])
                print(f"Exact Hit Accuracy: {acc:.4f}")

                accs.append(acc)


                if opt.related_position == 'begin':
                    for idx in range(opt.nuc):
                        print(f"Step {idx+1}:")
                        acc = calculate_exact_hit_accuracy([x[f'prediction_step_{idx+1}'] for x in list(generated_results.values())[1:]],
                                                        [x['target'] for x in list(generated_results.values())[1:]])
                        print(f"Exact Hit Accuracy: {acc:.4f}")
            print("accs:", accs)

        else:
            print(f"Running {dataset} dataset with nuc {opt.nuc}")
            
            if not os.path.exists(f"results/{dataset}"):
                os.makedirs(f"results/{dataset}", exist_ok=True)
            if not os.path.exists(f"results/{dataset}/{os.path.basename(opt.model)}"):
                os.makedirs(f"results/{dataset}/{os.path.basename(opt.model)}", exist_ok=True)
            interval_str = f"_interval_{opt.eval_interval}" if opt.eval_interval > 1 else ""
            filename = f"results/{dataset}/{os.path.basename(opt.model)}/results_samples_{opt.num_samples}_nuc_{opt.nuc}{interval_str}{'_backup' if opt.backup_memory else ''}_{opt.related_position}.json"

            is_cached = False
            if os.path.exists(filename):
                try:
                    cached_data = json.load(open(filename, 'r'))
                    if cached_data.get('param', {}).get('test_samples', 0) >= opt.num_samples:
                        generated_results = cached_data
                        is_cached = True
                    else:
                        print(f"Cached file {filename} has {cached_data.get('param', {}).get('test_samples', 0)} samples, which is fewer than requested {opt.num_samples}. Re-running evaluation...")
                except Exception:
                    pass

            if not is_cached:
                if opt.model is None:
                    model = None
                    tokenizer = None
                    middle_outputs, targets, contexts_middle, questions = run_qa(model, tokenizer, dataset, step=opt.nuc, output_filename=filename)
                    
                else:
                    if model is None or tokenizer is None:
                        model, tokenizer = load_model_and_tokenizer(opt.model, opt.split_model)

                    middle_outputs, targets, contexts_middle, questions = run_qa(model, tokenizer, dataset, step=opt.nuc, output_filename=filename)

                save_formatted_json(filename, opt, targets, middle_outputs, contexts_middle, questions)
                generated_results = json.load(open(filename, 'r'))
            
            print("Step 0 (0k tokens - No Distractors):")
            samples = [val for key, val in generated_results.items() if key.isdigit()]
            acc = calculate_exact_hit_accuracy([x['w/ context'] for x in samples],
                                            [x['target'] for x in samples])
            print(f"Exact Hit Accuracy: {acc:.4f}")

            if opt.related_position == 'begin' and len(samples) > 0:
                step_keys = sorted([k for k in samples[0].keys() if k.startswith('prediction_step_')],
                                   key=lambda x: int(x.replace('prediction_step_', '')))
                for step_key in step_keys:
                    step_num = int(step_key.replace('prediction_step_', ''))
                    tokens_approx = step_num * 512 / 1000
                    print(f"Step {step_num} (~{tokens_approx:.1f}k distractor tokens):")
                    acc = calculate_exact_hit_accuracy([x[step_key] for x in samples if step_key in x],
                                                    [x['target'] for x in samples if step_key in x])
                    print(f"Exact Hit Accuracy: {acc:.4f}")

