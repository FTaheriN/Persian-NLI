[mBERT](https://huggingface.co/bert-base-multilingual-cased) and [ParsBERT](https://github.com/hooshvare/parsbert) models are used to train Natural Language Inference on the [FarsTail](https://github.com/dml-qom/FarsTail) dataset. 
This model enables NLI in the Persian language. 

First, the data is preprocessed and the model is loaded. The model is then trained using cross-entropy as the loss function. Configuration to the model and dataset can be made via the config.yaml file. 

The repository also supports different inputs to the model (input_ids, attention_mask and token_type_ids) as well as using different outputs of the model to calculate the loss. 
Attention heatmap can also be plotted using the plot_attention function. 
