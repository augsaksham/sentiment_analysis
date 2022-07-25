import torch
import numpy as np
from utils import utils
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, RandomSampler
from torch.utils.data import DataLoader
import sys

device = torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('./saved_model/')

def __init__(**kwargs):
    pass

def predict(text,model_path,device=torch.device('cpu')):
    model_eval=utils.load_model(model_path)
    text=utils.clean_text(text)
    encoded_data_val = tokenizer.batch_encode_plus(
        np.array([text]),
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(np.array([1]))
    dataset_val = TensorDataset(input_ids_val, 
                            attention_masks_val,
                           labels_val)
    dataloader_val = DataLoader(
        dataset_val,
        sampler=RandomSampler(dataset_val),
        batch_size=1
    )
    predictions, true_vals = [], []
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0].type(torch.LongTensor),
                  'attention_mask': batch[1].type(torch.LongTensor),
                  'labels':         batch[2].type(torch.LongTensor),
                 }

        with torch.no_grad():        
            outputs = model_eval(**inputs)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    predictions = np.concatenate(predictions, axis=0)
    
    pred=np.argmax(predictions, axis=1).flatten()
    
    if(pred[0]==1):
        return "Positive"
    return "Negative"




