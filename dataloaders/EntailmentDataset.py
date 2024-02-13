import torch
import numpy as np
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader


class EntailmentDataset(Dataset):
    def __init__(self, dataframe, model_name):
        self.label_dict = {'e':0, 'c':1, 'n':2}
        
        self.df = dataframe

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.dataset = self.prep_data()
      

    def prep_data(self):
        input_ids = []
        token_type_ids = []
        attention_masks = []
        y = []

        for (prem, hypo, label) in zip(list(self.df['premise']), list(self.df['hypothesis']), list(self.df['label'])):

            encoded = self.tokenizer.encode_plus(text=prem, text_pair=hypo, return_tensors='pt',
                                                 max_length=140, pad_to_max_length=True, 
                                                 return_token_type_ids=True, return_attention_mask=True)


            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            token_type_ids.append(encoded['token_type_ids'])
            y.append(self.label_dict[label])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0) 
        token_type_ids = torch.cat(token_type_ids, dim=0) 
        y = torch.tensor(y)

        dataset = TensorDataset(input_ids, token_type_ids, attention_masks, y)
        print("dataset length: ", len(dataset))
        return dataset

    def get_data_loader(self, batch_size, shuffle=True):
        data_loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size)
        return data_loader