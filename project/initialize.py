#This file lodas he model architecture in memory for thwe first 
#It speeds up the inference time for subsequent model calls 
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
                                          'bert-base-uncased', 
                                          num_labels = 2,
                                          output_attentions = False,
                                          output_hidden_states = False
                                         )