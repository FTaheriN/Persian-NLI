[mBERT](https://huggingface.co/bert-base-multilingual-cased) and [ParsBERT](https://github.com/hooshvare/parsbert) models are used to train Natural Language Inference on the [FarsTail](https://github.com/dml-qom/FarsTail) dataset. 
This model enables NLI in the Persian language. 

First, the data is preprocessed and the model is loaded. The model is then trained using cross-entropy as the loss function. Configuration to the model and dataset can be made via the config.yaml file. 
