from torch import nn
import torch.nn.functional as F
from transformers import BertModel



class BertWithNNClassifier_1(nn.Module):
    
    def __init__(self, model_name, part, layer=-1):

        super(BertWithNNClassifier_1, self).__init__()

        self.part = part
        self.layer = layer
        self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()

        self.batch_norm1 = nn.BatchNorm1d(1024)
      
        self.linear1 = nn.Linear(768, 1024)
        self.linear2 = nn.Linear(1024, 3)

        for param in self.bert.named_parameters():
            param[1].requires_grad=F=True

        
    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, 
                                   attention_mask=attention_mask)

        if self.part == 1:
            output = bert_output.last_hidden_state[:,0,:]
        elif self.part == 2:
            output_ = bert_output.hidden_states[self.layer]
            output = output_[:,0,:]
        else:
            output = bert_output.pooler_output


        out = self.dropout1(output)

        out = self.linear1(out) #out
        out = self.relu(self.batch_norm1(out))
        out = self.linear2(self.dropout2(out))

        return out