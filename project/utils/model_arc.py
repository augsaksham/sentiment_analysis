#File for defining model architecute 
from transformers import BertForSequenceClassification


def get_model():
    model = BertForSequenceClassification.from_pretrained(
                                          'bert-base-uncased', 
                                          num_labels = 2,
                                          output_attentions = False,
                                          output_hidden_states = False
                                         )
    return model
