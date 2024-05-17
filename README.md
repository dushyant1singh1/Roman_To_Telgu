# Roman_To_Telgu
- In this assignment I am trying to make model predict the correct translitration from Roman to Telgu
### Instruction to run the code - 

- First you have to have the dataset provided in the assignment
- Take out the files which are for telgu dataset
- There are three files train , valid and test
- Put those three in the same folder where scripts are
- Don't make any kind of folder for these dataset
- Now you can run the scripts with default parameters
- To run
```python
python3 name_of_the_script.py
```

- Now we have similar arguments for both scripts and these are -
```python
"-wp","--wandb_project",default="myprojectname"
"-we","--wandb_entity",default="myname"
"-el","--encoder_layers",default=1,type=int
"-dl","--decoder_layers",default =1, type =int
"-emd","--embeded_dim",type = int, default=64
"-cell","--cell_type",default='rnn' , choices = ["rnn","gru","lstm"] , thses are for architecutre whether to choose from rnn, gru and lstm
"-bi","--bidirectional",default=False,type=bool
"-drop","--dropout",default=0,type=float
"-hn","--hidden_neurons",type =int, default=64
"--epochs","-e", help= "Number of epochs to train neural network", type= int, default=10
"--batch_size","-b",help="Batch size used to train neural network", type =int, default=1
"--optimizer","-o",help="batch size is used to train neural network", default= "adam", choices=["adam","nadam"]
"--learning_rate","-lr", default=0.0001, type=float
 
```

- This project have two scripts
  - one for seq2seq model without attention
  - second for with attention

- predicted_values.csv files contain all the predicted values with the best model using attention layers
- With attention model is giving 49% on validation dataset and on test dataset it is giving 46%
- Without attention model performs worse but still gives accuracy of 36%
