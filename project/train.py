#This file trains a Bert for text classification
#This file is called via command line eg. pythom train.py

from utils import utils
import torch
import numpy as np
import eval
import random
from transformers import AdamW, get_linear_schedule_with_warmup



def __init__(**kwargs):
    pass


#Initializer funtion to set seeds etc
def initialize():
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    

#Inputs ->
# data path :(path so the csv file)
# epochs :number of epochs to run the model
# save path :the path to store the trained model
# device :which device to use {gpu or cpu} default= cpu 
data = input("Input dataframe path ")
epoch = input("Input epochs ")
save_path = input("Input model save path ex -> save_model.pth")
device = input("Input device 'cpu' or 'gpu' ")
if device==None:
    device='cpu'
device=torch.device(device)
initialize()

def train_model(data,epochs,save_path,device=torch.device('cpu')):

    epochs=int(epochs)

    #Getting model class
    model=utils.get_model()

    #Getting dataloaders
    dataloader_train,dataloader_val =utils.get_data_loader(data)

    #Setting optimizer 
    optimizer = AdamW(
        model.parameters(),
        lr = 1e-5,
        eps = 1e-8
    )

    #Setting scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1,
        num_training_steps = len(dataloader_train)*epochs
    )

    #Main train loop
    for epoch in range(1, epochs+1):
        model.train()
        loss_train_total = 0

        print("Epoch : ",epoch)
        
        for i, batch in enumerate(dataloader_train):
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            
            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total +=loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            if i%1==0 :
                print("Batch : ",i)        
        loss_train_avg = loss_train_total/len(dataloader_train)
        
        
        val_loss, predictions, true_vals = eval.evaluate(dataloader_val)
        val_f1 = utils.f1_score_func(predictions, true_vals)
        

    #Saving the final model
    torch.save(model.state_dict(), save_path)
    print("Saved model")

train_model(data,epoch,save_path,device)    

