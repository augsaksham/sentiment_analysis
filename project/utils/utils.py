import re
import string
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split




from transformers import BertForSequenceClassification


def get_model():
    model = BertForSequenceClassification.from_pretrained(
                                          'bert-base-uncased', 
                                          num_labels = 2,
                                          output_attentions = False,
                                          output_hidden_states = False
                                         )
    return model

## Text Cleaning Functions ##
def remove_tag(text):
    ln=len(text)
    ind=-1
    for j in range(ln):
        if text[j]=='@':
            ind=j
        if ind==-1:
            return text
        else:
            in_sp=-1
            for i in range(ind,ln):
                if text[i]==' ':
                    in_sp=i
                    break
            if in_sp==-1:
                text=text[0:ind]
            else:
                
                text=text[0:ind]+text[in_sp+1:]
            ind=-1
                
        ln=len(text)
        if(ln<=j):
            break
        
    return text

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)

def remove_emojis(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def clean_text(text):
    text=remove_tag(text)
    rm_url=remove_URL(text)
    rm_html=remove_html(rm_url)
    rm_emoji=remove_emojis(rm_html)
    rm_punch=remove_punct(rm_emoji)
    
    return rm_punch

def fun_map(x):
    if x.lower()=='positive':
        return 1
    return 0

def get_data_loader(df_path):

    data_df=pd.read_csv(df_path)
    data_df['text']=data_df.text.map(lambda x: clean_text(x))
    data_df['airline_sentiment']=data_df.airline_sentiment.map(lambda x: fun_map(x))
    data_df=data_df.drop('Unnamed: 0',axis=1)
    data_df.reset_index(drop=True)
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    data_df=data_df.reset_index()
    data_df.rename(columns = {'airline_sentiment':'category', 'index':'id'}, inplace = True)
    data_df.set_index('id')
    df=data_df.copy()
    df=df.set_index('id')
    X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.category.values, 
                                                  test_size=0.15, 
                                                  random_state=42,
                                                  stratify=df.category.values)
    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )


    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].category.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].category.values) 

    dataset_train = TensorDataset(input_ids_train, 
                              attention_masks_train,
                              labels_train)

    dataset_val = TensorDataset(input_ids_val, 
                                attention_masks_val,
                            labels_val)

    batch_size = 4

    dataloader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=batch_size
    )

    dataloader_val = DataLoader(
        dataset_val,
        sampler=RandomSampler(dataset_val),
        batch_size=32
    )

    return dataloader_train,dataloader_val                                                          

def accuracy_per_class(preds, labels):
    label_dict={1:'1',0:'0'}
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')    

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')


def load_model(path):
    model_eval = get_model()
    model_eval.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    device = torch.device('cpu')
    model_eval.to(device)
    model_eval.eval()
    return model_eval               