iteration = 1
interval = 2500 #100 # 2500
dataset_name = "capitals" # "trivia_qa" #"capitals"
model_name = "open_llama_7b" #"opt-30b"
layer_number = -1 #7 #-1
start = iteration * interval
end = start + interval

import os
from datetime import datetime
from typing import Any, Dict
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict, Counter
from functools import partial
import re
from captum.attr import IntegratedGradients

gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# hardcode below,for now. Could dig into all models but they take a while to load
model_num_layers = {
    "falcon-40b" : 80,
    "falcon-7b" : 32,
    "open_llama_13b" : 40,
    "open_llama_7b" : 32,
    "opt-6.7b" : 32,
    "opt-30b" : 48,
}

assert layer_number < model_num_layers[model_name]
coll_str = "[0-9]+" if layer_number==-1 else str(layer_number)

model_repos = {
    "falcon-40b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
    "falcon-7b" : ("tiiuae", f".*transformer.h.{coll_str}.mlp.dense_4h_to_h", f".*transformer.h.{coll_str}.self_attention.dense"),
    "open_llama_13b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
    "open_llama_7b" : ("openlm-research", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
    "opt-6.7b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj"),
    "opt-30b" : ("facebook", f".*model.decoder.layers.{coll_str}.fc2", f".*model.decoder.layers.{coll_str}.self_attn.out_proj", ),
}

dataset_names = ["capitals", "place_of_birth", "trivia_qa", "founders"]

if dataset_name in ["capitals", "place_of_birth", "founders"]:
    pd_frame = pd.read_csv(f'/home/ec2-user/SageMaker/halu_code/data/{dataset_name}.csv')
    dataset = [(pd_frame.iloc[i]['subject'], pd_frame.iloc[i]['target']) for i in range(start, end)]
elif dataset_name=="trivia_qa":
    trivia_qa = load_dataset('trivia_qa', 'rc.nocontext', cache_dir='/home/ec2-user/SageMaker/halu_code/cache/data')
    full_dataset = []
    for obs in tqdm(trivia_qa['train']):
        aliases = []
        aliases.extend(obs['answer']['aliases'])
        aliases.extend(obs['answer']['normalized_aliases'])
        aliases.append(obs['answer']['value'])
        aliases.append(obs['answer']['normalized_value'])
        full_dataset.append((obs['question'], aliases))
    dataset = full_dataset[start: end]

model_loader = AutoModelForSeq2SeqLM if "t5" in model_name \
                   else LlamaForCausalLM if "llama" in model_name \
                   else AutoModelForCausalLM

token_loader = LlamaTokenizer if "llama" in model_name \
               else AutoTokenizer

tokenizer = token_loader.from_pretrained(f'{model_repos[model_name][0]}/{model_name}')

model = model_loader.from_pretrained(f'{model_repos[model_name][0]}/{model_name}', cache_dir="/home/ec2-user/SageMaker/halu_code/cache/models", 
                                             device_map="auto", 
                                             torch_dtype=torch.bfloat16,
                                             #load_in_4bit=True,
                                             trust_remote_code=True)

fully_connected_hidden_layers = defaultdict(list)

def save_fully_connected_hidden(name, mod, inp, out):
    fully_connected_hidden_layers[name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())

fully_connected_forward_handles = {}

for name, module in model.named_modules():
    if re.match(f'{model_repos[model_name][1]}$', name):
        fully_connected_forward_handles[name] = module.register_forward_hook(partial(save_fully_connected_hidden, name))
        
attention_hidden_layers = defaultdict(list)

def save_attention_hidden(name, mod, inp, out):
    attention_hidden_layers[name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())

attention_forward_handles = {}

for name, module in model.named_modules():
    if re.match(f'{model_repos[model_name][2]}$', name):
        attention_forward_handles[name] = module.register_forward_hook(partial(save_attention_hidden, name))
        

if "t5" in model_name:
    stop_tokens = 1
elif "llama" in model_name:
    stop_tokens = 13
elif "falcon" in model_name:
    stop_tokens = 193
else:
    stop_tokens = 50118
    
def get_next_token(x):
    with torch.no_grad():
        return model(x).logits

def get_next_token_t5(encoder_input_ids, decoder_input_ids):
    with torch.no_grad():
        return model(encoder_input_ids, decoder_input_ids=decoder_input_ids).logits

def generate_response(x, max_length=100, pbar=False):
    response = []
    bar = tqdm(range(max_length)) if pbar else range(max_length)
    for step in bar:
        logits = get_next_token(x)
        next_token = logits.squeeze()[-1].argmax()
        x = torch.concat([x, next_token.view(1, -1)], dim=1)
        response.append(next_token)
        if next_token == stop_tokens and step>5:
            break
    return torch.stack(response).cpu().numpy(), logits.squeeze()

def generate_response_t5(encoder_input_ids, max_length=100, pbar=False):
    response = []
    bar = tqdm(range(max_length)) if pbar else range(max_length)
    decoder_input_ids = torch.tensor([[0,3]]).to(device)
    for step in bar:
        logits = get_next_token_t5(encoder_input_ids, decoder_input_ids)
        next_token = logits.squeeze()[-1].argmax()
        decoder_input_ids = torch.concat([decoder_input_ids, next_token.view(1, -1)], dim=1)
        response.append(next_token)
        if next_token == 1:
            break
    return torch.stack(response).cpu().numpy(), logits.squeeze()[-1]

response_generator = generate_response_t5 if "t5" in model_name else generate_response

def answer_question(question, max_length=100, pbar=False):
    input_ids = tokenizer(question, return_tensors='pt').input_ids.to(device)
    response, logits = response_generator(input_ids, max_length=max_length, pbar=pbar)
    return response, logits, input_ids.shape[-1]

def answer_trivia(question, targets):
    response, logits, start_pos = answer_question(question)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = False
    for alias in targets:
        if alias.lower() in str_response.lower():
            correct = True
            break
    return response, str_response, logits, start_pos, correct

def answer_capitals(source, target):
    question = f"What is the capital of {source}?"
    response, logits, start_pos = answer_question(question)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = False
    if target.lower() in str_response.lower():
        correct = True
    return response, str_response, logits, start_pos, correct

def answer_birth_place(source, target):
    question = f"Where was {source} born?"
    response, logits, start_pos = answer_question(question)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = False
    if target.lower() in str_response.lower():
        correct = True
    return response, str_response, logits, start_pos, correct

def answer_founders(source, target):
    question = f"Who founded {source}?"
    response, logits, start_pos = answer_question(question)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = False
    if target.lower() in str_response.lower():
        correct = True
    return response, str_response, logits, start_pos, correct

def collect_fully_connected(token_pos):
    layer_name = model_repos[model_name][1][2:].split(coll_str)
    if "t5" in model_name:
        layer_count = model.decoder.block
    elif "llama" in model_name:
        layer_count = model.model.layers
    elif "falcon" in model_name:
        layer_count = model.transformer.h
    else:
        layer_count = model.model.decoder.layers
        
    layer_st = 0 if layer_number==-1 else layer_number
    layer_en = len(layer_count) if layer_number==-1 else layer_number+1
    first_activation = np.stack([fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_st, layer_en)])
    final_activation = np.stack([fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                for i in range(layer_st, layer_en)])
    return first_activation, final_activation

def collect_attention(token_pos):
    layer_name = model_repos[model_name][2][2:].split(coll_str)
    if "t5" in model_name:
        layer_count = model.decoder.block
    elif "llama" in model_name:
        layer_count = model.model.layers
    elif "falcon" in model_name:
        layer_count = model.transformer.h
    else:
        layer_count = model.model.decoder.layers
        
    layer_st = 0 if layer_number==-1 else layer_number
    layer_en = len(layer_count) if layer_number==-1 else layer_number+1
    first_activation = np.stack([attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_st, layer_en)])
    final_activation = np.stack([attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                for i in range(layer_st, layer_en)])
    return first_activation, final_activation

def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
        # attributes has shape (batch, sequence size, embedding dim)
        attributes = attributes.squeeze(0)

        # if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
        norm = torch.norm(attributes, dim=1)
        attributes = norm / torch.sum(norm)  # Normalize the values so they add up to 1
        
        return attributes

def model_forward(input_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
            -> torch.Tensor:
        output = model(inputs_embeds=input_, **extra_forward_args)
        return torch.nn.functional.softmax(output.logits[:, -1, :], dim=-1)

forward_func = partial(model_forward, model=model, extra_forward_args={})

ig_steps = 64
internal_batch_size = 4
attr_method = IntegratedGradients

def get_ig(prompt):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    prediction_id = get_next_token(input_ids).squeeze()[-1].argmax()
    if "falcon" in model_name:
        embedder = model.transformer.word_embeddings
    elif "opt" in model_name:
        embedder = model.model.decoder.embed_tokens
    elif "llama" in model_name:
        embedder = model.model.embed_tokens
    encoder_input_embeds = embedder(input_ids).detach() # fix this for each model
    ig = attr_method(forward_func=forward_func)
    attributes = normalize_attributes(ig.attribute(encoder_input_embeds, 
                                                              target=prediction_id, 
                                                              n_steps=ig_steps, 
                                                              internal_batch_size=internal_batch_size)).detach().cpu().numpy()
    return attributes

results = defaultdict(list)

if dataset_name=="capitals":
    question_asker = answer_capitals
elif dataset_name=="trivia_qa":
    question_asker = answer_trivia
elif dataset_name=="place_of_birth":
    question_asker = answer_birth_place
elif dataset_name=="founders":
    question_asker = answer_founders

for idx in tqdm(range(len(dataset))): 
    fully_connected_hidden_layers.clear()
    attention_hidden_layers.clear()
    
    question, answers = dataset[idx]
    response, str_response, logits, start_pos, correct = question_asker(question, answers)
    first_fully_connected, final_fully_connected = collect_fully_connected(start_pos)
    first_attention, final_attention = collect_attention(start_pos)
    attributes_first = get_ig(question)

    results['question'].append(question)
    results['answers'].append(answers)
    results['response'].append(response)
    results['str_response'].append(str_response)
    results['logits'].append(logits.to(torch.float32).cpu().numpy())
    results['start_pos'].append(start_pos)
    results['correct'].append(correct)
    results['first_fully_connected'].append(first_fully_connected)
    results['final_fully_connected'].append(final_fully_connected)
    results['first_attention'].append(first_attention)
    results['final_attention'].append(final_attention)
    results['attributes_first'].append(attributes_first)
    
with open(f"/home/ec2-user/SageMaker/halu_code/results/{model_name}_{dataset_name}_{datetime.now().month}_{datetime.now().day}.pickle", "wb") as outfile:
    outfile.write(pickle.dumps(results))
