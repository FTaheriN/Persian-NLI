import time
import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from dataloaders import EntailmentDataset
from deeplearning.train import *
from deeplearning.test import *
from model import BertWithNNClassifier_1, BertWithNNClassifier
from utils import *

############################## Reading Model Parameters ##############################
config = read_yaml_config()
train_path = config['train_path']
valid_path = config['valid_path']
test_path  = config['test_path' ]

epochs = config['epochs']
batch_size = config['batch_size']

model = config['model']
part  = config['part' ]
layer = config['layer']
bert  = config['bert' ]
model_pth = config['model_pth']

section = config['section']

####################################      Main     #################################### 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_df = pd.read_csv("/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/data/Train-word-proc.csv")
    valid_df = pd.read_csv("/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/data/Valid-word-proc.csv")
    test_df  = pd.read_csv("/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/data/Test-word-proc.csv" )

    train_pars_bert = EntailmentDataset(train_df, bert)
    valid_pars_bert = EntailmentDataset(valid_df, bert)
    test_pars_bert  = EntailmentDataset(test_df , bert)

    train_loader = train_pars_bert.get_data_loader(batch_size=batch_size)
    valid_loader = valid_pars_bert.get_data_loader(batch_size=batch_size)
    test_loader  =  test_pars_bert.get_data_loader(batch_size=batch_size) 

    if section != 7:
        bert_model = BertWithNNClassifier(bert, part, layer)
    else: 
        bert_model = BertWithNNClassifier_1(bert, part, layer)

    optimizer = torch.optim.Adam(bert_model.parameters(), lr=0.00002)
    lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    train_loss_1, train_acc_1, valid_loss_1, valid_acc_1 = train(bert_model, 
                                                         train_loader, valid_loader, 
                                                         criterion, optimizer, lr_scheduler,
                                                         model_pth, device)
    
    report_path = "/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/report/" + model_pth
    save_report(train_loss_1, valid_loss_1, train_acc_1, valid_acc_1, report_path)
    plot_loss_acc(report_path)
    
    model = torch.load("/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/checkpoints/" + model_pth)
    test_acc_, preds_, y_test = test(test_loader, model, device, criterion)
    test_report(y_test, preds_)

    plot_attention(test_df)


main()