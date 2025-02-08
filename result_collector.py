import functools
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
from string import Template


# Data related params
iteration = 0
interval = 2500 # We run the inference on these many examples at a time to achieve parallelization
start = iteration * interval
end = start + interval
dataset_name =  "place_of_birth" # "trivia_qa" #"capitals"
trex_data_to_question_template = {
    "capitals": Template("What is the capital of $source?"),
    "place_of_birth": Template("Where was $source born?"),
    "founders": Template("Who founded $source?"),
}

# IO
data_dir = Path(".") # Where our data files are stored
model_dir = Path("./.cache/models/") # Cache for huggingface models
results_dir = Path("./results/") # Directory for storing results

# Hardware
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# Integrated Grads
ig_steps = 64
internal_batch_size = 4

# Model
model_name = "falcon-7b" #"opt-30b"
layer_number = -1
# hardcode below,for now. Could dig into all models but they take a while to load
model_num_layers = {
    "falcon-40b" : 60,
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

# For storing results
fully_connected_hidden_layers = defaultdict(list)
attention_hidden_layers = defaultdict(list)
attention_forward_handles = {}
fully_connected_forward_handles = {}


def save_fully_connected_hidden(layer_name, mod, inp, out):
    fully_connected_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


def save_attention_hidden(layer_name, mod, inp, out):
    attention_hidden_layers[layer_name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())


def get_stop_token():
    if "llama" in model_name:
        stop_token = 13
    elif "falcon" in model_name:
        stop_token = 193
    else:
        stop_token = 50118
    return stop_token


def load_data(dataset_name):
    if dataset_name in trex_data_to_question_template.keys():
        pd_frame = pd.read_csv(data_dir / f'{dataset_name}.csv')
        dataset = [(pd_frame.iloc[i]['subject'], pd_frame.iloc[i]['object'].split("<OR>")) for i in range(start, min(end, len(pd_frame)))]
    elif dataset_name=="trivia_qa":
        trivia_qa = load_dataset('trivia_qa', data_dir='rc.nocontext', cache_dir=str(data_dir))
        full_dataset = []
        for obs in tqdm(trivia_qa['train']):
            aliases = []
            aliases.extend(obs['answer']['aliases'])
            aliases.extend(obs['answer']['normalized_aliases'])
            aliases.append(obs['answer']['value'])
            aliases.append(obs['answer']['normalized_value'])
            full_dataset.append((obs['question'], aliases))
        dataset = full_dataset[start: end]
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    return dataset


def get_next_token(x, model):
    with torch.no_grad():
        return model(x).logits


def generate_response(x, model, *, max_length=100, pbar=False):
    response = []
    bar = tqdm(range(max_length)) if pbar else range(max_length)
    for step in bar:
        logits = get_next_token(x, model)
        next_token = logits.squeeze()[-1].argmax()
        x = torch.concat([x, next_token.view(1, -1)], dim=1)
        response.append(next_token)
        if next_token == get_stop_token() and step>5:
            break
    return torch.stack(response).cpu().numpy(), logits.squeeze()


def answer_question(question, model, tokenizer, *, max_length=100, pbar=False):
    input_ids = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
    response, logits = generate_response(input_ids, model, max_length=max_length, pbar=pbar)
    return response, logits, input_ids.shape[-1]


def answer_trivia(question, targets, model, tokenizer):
    response, logits, start_pos = answer_question(question, model, tokenizer)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = False
    for alias in targets:
        if alias.lower() in str_response.lower():
            correct = True
            break
    return response, str_response, logits, start_pos, correct


def answer_trex(source, targets, model, tokenizer, question_template):
    response, logits, start_pos = answer_question(question_template.substitute(source=source), model, tokenizer)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    correct = any([target.lower() in str_response.lower() for target in targets])
    return response, str_response, logits, start_pos, correct


def get_start_end_layer(model):
    if "llama" in model_name:
        layer_count = model.model.layers
    elif "falcon" in model_name:
        layer_count = model.transformer.h
    else:
        layer_count = model.model.decoder.layers
    layer_st = 0 if layer_number == -1 else layer_number
    layer_en = len(layer_count) if layer_number == -1 else layer_number + 1
    return layer_st, layer_en


def collect_fully_connected(token_pos, layer_start, layer_end):
    layer_name = model_repos[model_name][1][2:].split(coll_str)
    first_activation = np.stack([fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_start, layer_end)])
    final_activation = np.stack([fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                for i in range(layer_start, layer_end)])
    return first_activation, final_activation


def collect_attention(token_pos, layer_start, layer_end):
    layer_name = model_repos[model_name][2][2:].split(coll_str)
    first_activation = np.stack([attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][token_pos] \
                                for i in range(layer_start, layer_end)])
    final_activation = np.stack([attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][-1][-1] \
                                for i in range(layer_start, layer_end)])
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


def get_embedder(model):
    if "falcon" in model_name:
        return model.transformer.word_embeddings
    elif "opt" in model_name:
        return model.model.decoder.embed_tokens
    elif "llama" in model_name:
        return model.model.embed_tokens
    else:
        raise ValueError(f"Unknown model {model_name}")

def get_ig(prompt, forward_func, tokenizer, embedder, model):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    prediction_id = get_next_token(input_ids, model).squeeze()[-1].argmax()
    encoder_input_embeds = embedder(input_ids).detach() # fix this for each model
    ig = IntegratedGradients(forward_func=forward_func)
    attributes = normalize_attributes(
        ig.attribute(
            encoder_input_embeds,
            target=prediction_id,
            n_steps=ig_steps,
            internal_batch_size=internal_batch_size
        )
    ).detach().cpu().numpy()
    return attributes


def compute_and_save_results():

    # Dataset
    dataset = load_data(dataset_name)
    if dataset_name in trex_data_to_question_template.keys():
        question_asker = functools.partial(answer_trex, question_template=trex_data_to_question_template[dataset_name])
    elif dataset_name == "trivia_qa":
        question_asker = answer_trivia
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}.")

    # Model
    model_loader = LlamaForCausalLM if "llama" in model_name else AutoModelForCausalLM
    token_loader = LlamaTokenizer if "llama" in model_name else AutoTokenizer
    tokenizer = token_loader.from_pretrained(f'{model_repos[model_name][0]}/{model_name}')
    model = model_loader.from_pretrained(f'{model_repos[model_name][0]}/{model_name}',
                                         cache_dir=model_dir,
                                         device_map=device,
                                         torch_dtype=torch.bfloat16,
                                         # load_in_4bit=True,
                                         trust_remote_code=True)
    forward_func = partial(model_forward, model=model, extra_forward_args={})
    embedder = get_embedder(model)

    # Prepare to save the internal states
    for name, module in model.named_modules():
        if re.match(f'{model_repos[model_name][1]}$', name):
            fully_connected_forward_handles[name] = module.register_forward_hook(
                partial(save_fully_connected_hidden, name))
        if re.match(f'{model_repos[model_name][2]}$', name):
            attention_forward_handles[name] = module.register_forward_hook(partial(save_attention_hidden, name))

    # Generate results
    results = defaultdict(list)
    for idx in tqdm(range(len(dataset))):
        fully_connected_hidden_layers.clear()
        attention_hidden_layers.clear()

        question, answers = dataset[idx]
        response, str_response, logits, start_pos, correct = question_asker(question, answers, model, tokenizer)
        layer_start, layer_end = get_start_end_layer(model)
        first_fully_connected, final_fully_connected = collect_fully_connected(start_pos, layer_start, layer_end)
        first_attention, final_attention = collect_attention(start_pos, layer_start, layer_end)
        attributes_first = get_ig(question, forward_func, tokenizer, embedder, model)

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
    with open(results_dir/f"{model_name}_{dataset_name}_start-{start}_end-{end}_{datetime.now().month}_{datetime.now().day}.pickle", "wb") as outfile:
        outfile.write(pickle.dumps(results))


if __name__ == '__main__':
    compute_and_save_results()