import pandas as pd
from preprocess import *


def read_file(file_path):
    return pd.read_csv(file_path, sep='\t', on_bad_lines='skip')


train_path = "/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/data/Train-word.csv"
valid_path = "/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/data/Valid-word.csv"
test_path  = "/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/data/Test-word.csv" 

###################################   Read Dataframes   ####################################
train_df = read_file(train_path)
valid_df = read_file(valid_path)
test_df  = read_file(test_path )

###################################   Plot Statistics   ####################################
plot_statistics(train_df, valid_df, test_df)

###################################   Keep Data <= 80   ####################################
train_df = train_df.loc[train_df['p_lengths'] <= 80].reset_index(drop=True)
valid_df = valid_df.loc[valid_df['p_lengths'] <= 80].reset_index(drop=True)
test_df  = test_df.loc [test_df ['p_lengths'] <= 80].reset_index(drop=True)

###################################   Preprocess Data   ####################################
                      #################   Optional   ##################
train_df, y_train = preprocess_data(train_df)
valid_df, y_valid = preprocess_data(valid_df)
test_df , y_test  = preprocess_data(test_df )

train_df.to_csv("/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/data/Train-word-proc.csv")  
valid_df.to_csv("/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/data/Valid-word-proc.csv")  
test_df.to_csv ("/content/drive/MyDrive/MSC/DeepLearning/HW4/Q1/data/Test-word-proc.csv" )   