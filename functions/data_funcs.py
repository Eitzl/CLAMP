import numpy as np
import random
import torch
import torch.nn as nn
# from rdkit import Chem


def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def reset_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def random_mask(input, tokenizer, max_length):

    # Encode input with special tokens and padding
    y_true = tokenizer(input, add_special_tokens=True, max_length=max_length, padding='max_length', 
                        truncation=True, return_tensors='pt').input_ids
    attention_mask = tokenizer(input, add_special_tokens=True, max_length=max_length, padding='max_length', 
                        truncation=True, return_tensors='pt').attention_mask

    # Encode input without special tokens or padding
    masked_input = tokenizer.encode(input, add_special_tokens=False, max_length=max_length-2, truncation=True)

    # randomly select indices to mask at 15% of the masked_input positions
    index_to_mask = random.sample(range(len(masked_input)), max(1, int(0.15 * len(masked_input))))

    # sort indices to mask in ascending order
    index_to_mask.sort()
    
    # Randomly select mask, old, or random
    mask_type = np.random.choice(['mask', 'old', 'random'], p=[0.8, 0.1, 0.1])
    
    # Loop through indices to mask and replace with mask, old, or random
    for index in index_to_mask:
        if mask_type == 'mask':
            masked_input[index] = tokenizer.mask_token_id
        elif mask_type == 'random':
            masked_input[index] = random.choice(range(5, tokenizer.vocab_size))

    # add bos, eos, and padding to max_length to masked input
    masked_input = tokenizer.encode(tokenizer.decode(masked_input), add_special_tokens=True, max_length=max_length, padding='max_length', 
                                    truncation=True, return_tensors='pt')

    # add one to all items in index_to_mask to account for BOS token
    index_to_mask = [i + 1 for i in index_to_mask]

    # generate single dimension tensor of indices to mask
    mask_idx = torch.zeros((max_length, ), dtype=torch.long).scatter_(0, torch.tensor(index_to_mask), 1)

    y_true = y_true.squeeze()
    
    # Set unmasked positions in y_true to -100
    y_true[mask_idx == 0] = -100
    
    return masked_input.squeeze(), y_true, attention_mask.squeeze() #, mask_idx
