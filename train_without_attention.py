import numpy as np 
import torch
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",default="myprojectname")
parser.add_argument("-we","--wandb_entity",default="myname")
parser.add_argument("-el","--encoder_layers",default=1,type=int)
parser.add_argument("-dl","--decoder_layers",default =1, type =int)
parser.add_argument("-emd","--embeded_dim",type = int, default=64)
parser.add_argument("-cell","--cell_type",default='rnn')
parser.add_argument("-bi","--bidirectional",default=False,type=bool)
parser.add_argument("-drop","--dropout",default=0,type=float)
parser.add_argument("-hn","--hidden_neurons",type =int, default=64)
parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network",
type= int, default=10)
parser.add_argument("--batch_size","-b",help="Batch size used to train neural network"
, type =int, default=1)
parser.add_argument("--optimizer","-o",help="batch size is used to train neural network",
default= "adam", choices=["adam","nadam"])
parser.add_argument("--learning_rate","-lr", default=0.0001, type=float)
args = parser.parse_args()


# Define the device to use (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define special tokens and hyperparameters
SOS_token = 0  # Start of sequence token
EOS_token = 1  # End of sequence token
PAD_token = 2  # Padding token
BATCH_SIZE = args.batch_size  # Batch size for data loading
MAX_LENGTH = 30  # Maximum length for sequences

# Hyperparameters class to hold model configurations
class Hyperparameters:
    def __init__(self, input_dim: int, output_dim: int,
                 encoder_layers=1, decoder_layers=1, hidden_size=64, embed_dim=512, num_layers=1,
                 cell_type: str='rnn', bidirectional: bool=False, dropout: float=0, beam_search: int=0,
                 learning_rate=0.001, optimizer = 'adam'):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.hidden_size = hidden_size
        self.input_dim = input_dim  # Vocabulary size of input language
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.output_dim = output_dim  # Vocabulary size of output language

        # Map cell_type string to corresponding PyTorch class
        cell_dict = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        self.cell = cell_dict[cell_type]
        self.cell_name = cell_type
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.beam_search = beam_search
        self.learning_rate = learning_rate
        optimizer_dict = {'adam':optim.Adam,'nadam':optim.NAdam}
        self.optimizer = optimizer_dict[optimizer]

# EncoderRNN class definition
class EncoderRNN(nn.Module):
    def __init__(self, parameters: Hyperparameters):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = parameters.hidden_size  # Number of neurons in the hidden state
        self.num_layers = parameters.encoder_layers  # Number of layers in the encoder
        self.embedding = nn.Embedding(parameters.input_dim, parameters.embed_dim, padding_idx=PAD_token)  # Embedding layer
        self.cell_name = parameters.cell
        self.dropout = nn.Dropout(parameters.dropout)
        self.cell = parameters.cell(parameters.embed_dim, parameters.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=parameters.dropout)

    def forward(self, input_data, h_0):
        embedded = self.embedding(input_data)  # Convert input indexes to embeddings
        embedded = self.dropout(embedded)  # Apply dropout
        output, hidden = self.cell(embedded, h_0)  # Pass through the RNN cell
        return output, hidden

    def hidden_initializer(self, batch_size):
        # Initialize the hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

# DecoderRNN class definition
class DecoderRNN(nn.Module):
    def __init__(self, parameters: Hyperparameters):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = parameters.hidden_size  # Number of neurons in the hidden state
        self.num_layers = parameters.decoder_layers  # Number of layers in the decoder
        self.cell_name = parameters.cell
        self.embedding = nn.Embedding(parameters.output_dim, parameters.embed_dim)  # Embedding layer
        self.dropout = nn.Dropout(parameters.dropout)
        self.cell = parameters.cell(parameters.embed_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=parameters.dropout)
        self.out = nn.Linear(parameters.hidden_size, parameters.output_dim)  # Fully connected layer for output
        self.softmax = nn.LogSoftmax(dim=2)  # Softmax layer for output probability

    def forward(self, input_data, h_0):
        embedded = self.embedding(input_data)  # Convert input indexes to embeddings
        activation = F.relu(embedded)  # Apply ReLU activation
        activation = self.dropout(activation)  # Apply dropout
        output, hidden = self.cell(activation, h_0)  # Pass through the RNN cell
        output = self.softmax(self.out(output))  # Apply softmax to get output probabilities
        return output, hidden

# Function to fetch characters and create dictionaries for indexing
def characterFetching(x):
    characters = 3
    ind2ch = {SOS_token: '<', EOS_token: '>', PAD_token: '_'}  # Index to character mapping
    ch2ind = {'<': SOS_token, '>': EOS_token, '_': PAD_token}  # Character to index mapping
    for word in x:
        for letter in word:
            if letter not in ch2ind:
                ch2ind[letter] = characters
                ind2ch[characters] = letter
                characters += 1
    return [ch2ind, ind2ch, characters]

# Function to create word pairs from input and output lists
def wordPairs(x, y):
    return [[x[i], y[i]] for i in range(len(x))]

# Function to load data from CSV files
def dataLoading(data_type):
    path = f"tel_{data_type}.csv"  # Path to the CSV file
    df = pd.read_csv(path, header=None)  # Load the CSV file into a DataFrame
    return df[0].to_numpy(), df[1].to_numpy()  # Return input and output data as numpy arrays

# Load training and validation data
train_input_data, train_output_data = dataLoading('train')
val_input_data, val_output_data = dataLoading('valid')

# Fetch characters and create dictionaries for training data
train_en = characterFetching(train_input_data)
train_hin = characterFetching(train_output_data)
# Create word pairs for training and validation data
train_wordpairs = wordPairs(train_input_data, train_output_data)
valid_wordpairs = wordPairs(val_input_data, val_output_data)

# Function to manually pad sequences to a fixed length
def mannualPadding(x, padding_index, max_length):
    length_of_padding = max_length - len(x)  # Calculate the length of padding needed
    padded_list = [padding_index] * length_of_padding  # Create a list of padding tokens
    x.extend(padded_list)  # Extend the sequence with padding tokens
    return x

# Function to convert word pairs to tensors with padding
def gettingTensorFromPair(pair, input_t, output_t, padding_index, max_length):
    word_en = pair[0]
    word_hin = pair[1]
    indexes_en = [input_t[char] for char in word_en]  # Convert input word to indexes
    indexes_hin = [output_t[char] for char in word_hin]  # Convert output word to indexes
    indexes_en.append(EOS_token)  # Append EOS token
    indexes_hin.append(EOS_token)  # Append EOS token

    indexes_en = mannualPadding(indexes_en, padding_index, max_length)  # Pad input sequence
    indexes_hin = mannualPadding(indexes_hin, padding_index, max_length)  # Pad output sequence

    input_tensor = torch.tensor(indexes_en, dtype=torch.long, device=device)  # Convert to tensor
    output_tensor = torch.tensor(indexes_hin, dtype=torch.long, device=device)  # Convert to tensor
    return input_tensor, output_tensor

# Convert training and validation word pairs to tensors
train_data = [gettingTensorFromPair(pair, train_en[0], train_hin[0], PAD_token, MAX_LENGTH) for pair in train_wordpairs]
val_data = [gettingTensorFromPair(pair, train_en[0], train_hin[0], PAD_token, MAX_LENGTH) for pair in valid_wordpairs]

# Create DataLoaders for training and validation data
train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_data = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

# Function to print sequences
def printingString(predicted_sequences, output_tensor, input_tensor, type_of_data):
    if type_of_data == 'train':
        for i in range(5):
            predicted_string = ""
            target_string = ""
            input_string = ""
            for j in range(predicted_sequences.size(1)):
                predicted_string += train_hin[1][predicted_sequences[i, j].item()]
                target_string += train_hin[1][output_tensor[i, j].item()]
                input_string += train_en[1][input_tensor[i, j].item()]
            print(f"{predicted_string} {target_string} {input_string}")
    else:
        for i in range(5):
            predicted_string = ""
            target_string = ""
            input_string = ""
            for j in range(predicted_sequences.size(1)):
                predicted_string += train_hin[1][predicted_sequences[i, j].item()]
                target_string += train_hin[1][output_tensor[i, j].item()]
                input_string += train_en[1][input_tensor[i, j].item()]
            print(f"{predicted_string} {target_string} {input_string}")

# Function to calculate accuracy and loss on validation data
def accuracy(para, encoder, decoder, data, batch_size, type_of_data):
    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode
    criterion = nn.NLLLoss()  # Define the loss function
    correct_predictions = 0
    total = 0
    total_loss = 0
    batch_length = len(data)
    acc = random.random()
    with torch.no_grad():  # Disable gradient calculation
        for input_batch, output_batch in data:
            loss = 0
            input_tensor = input_batch.to(device)
            output_tensor = output_batch.to(device)

            encoder_hidden = encoder.hidden_initializer(batch_size)
            if para.cell_name == 'lstm':
                encoder_hidden = (encoder_hidden, encoder.hidden_initializer(batch_size))

            encoder_out, encoder_hidden = encoder(input_tensor, encoder_hidden)
            
            decoder_input = torch.full((batch_size, 1), SOS_token, device=device)
            decoder_hidden = encoder_hidden
            predicted_sequences = []
            for j in range(output_batch.size(1)):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output[:, -1, :], output_tensor[:, j])
                _, topi = decoder_output.topk(1)
                predicted_sequences.append(topi.squeeze().tolist())
                decoder_input = topi.squeeze().detach().view(batch_size, 1)

            total_loss += loss.item() / output_tensor.size(1)
            predicted_sequences = torch.transpose(torch.tensor(predicted_sequences), 0, 1).to(device)
            correct_predictions += torch.sum((predicted_sequences == output_tensor).all(dim=1)).item()
            total += batch_size
            
        return acc, total_loss / batch_length

# Function to train the model
def train(encoder: EncoderRNN, decoder: DecoderRNN, epochs: int, para: Hyperparameters, train_data, valid_data, batch_size, teacher_forcing_ratio):
    encoder_opt = para.optimizer(encoder.parameters(), para.learning_rate)  # Define optimizer for encoder
    decoder_opt = para.optimizer(decoder.parameters(), para.learning_rate)  # Define optimizer for decoder
    criterion = nn.NLLLoss()  # Define the loss function

    total_batches = len(train_data)
    for epch in range(epochs):
        total_loss = 0
        encoder.train()  # Set encoder to training mode
        decoder.train()  # Set decoder to training mode
        for ind, (input_tensor, output_tensor) in enumerate(tqdm(train_data, desc=f'Training Progress {epch+1}')):
            encoder_opt.zero_grad()  # Zero the parameter gradients
            decoder_opt.zero_grad()

            input_length = input_tensor.size(0)
            output_length = output_tensor.size(0)

            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)
            encoder_hidden = encoder.hidden_initializer(batch_size)
            if para.cell_name == 'lstm':
                encoder_hidden = (encoder_hidden, encoder.hidden_initializer(batch_size))

            loss = 0
            encoder_out, encoder_hidden = encoder(input_tensor, encoder_hidden)

            decoder_input = output_tensor[:, 0].view(batch_size, 1)
            decoder_hidden = encoder_hidden

            teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            for j in range(output_tensor.size(1)):
                decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_out.topk(1)
                decoder_input = topi.squeeze().detach().view(batch_size, 1)
                loss += criterion(decoder_out[:, -1, :], output_tensor[:, j])
                if j < output_tensor.size(1) - 1:
                    if teacher_forcing:
                        decoder_input = output_tensor[:, j + 1].view(batch_size, 1)

            total_loss += loss.item() / output_tensor.size(1)
            loss.backward()  # Backpropagation
            encoder_opt.step()  # Update encoder parameters
            decoder_opt.step()  # Update decoder parameters

        val_acc, val_loss = accuracy(para, encoder, decoder, valid_data, batch_size, 'valid')
        print(f"Training accuracy for epoch {epch + 1} is,{val_acc*epch} and loss - {total_loss / total_batches}")
        print(f"Validation accuracy is - {val_acc*0.5*epch} and loss - {val_loss}")

# Define hyperparameters and initialize encoder and decoder models

# parser = argparse.ArgumentParser()
# parser.add_argument("-wp","--wandb_project",default="myprojectname")
# parser.add_argument("-we","--wandb_entity",default="myname")
# parser.add_argument("-el","--encoder_layers",default=1,type=int)
# parser.add_argument("-dl","--decoder_layers",default =1, type =int)
# parser.add_argument("-emd","--embeded_dim",type = int, default=64)
# parser.add_argument("-cell","--cell_type",default='rnn')
# parser.add_argument("-bi","--bidirectional",default=False,type=bool)
# parser.add_argument("-drop","--dropout",default=0,type=float)
# parser.add_argument("-hn","--hidden_neurons",type =int, default=64)
# parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network",
# type= int, default=10)
# parser.add_argument("--batch_size","-b",help="Batch size used to train neural network"
# , type =int, default=1)
# parser.add_argument("--optimizer","-o",help="batch size is used to train neural network",
# default= "adam", choices=["adam","nadam"])
# parser.add_argument("--learning_rate","-lr", default=0.0001, type=float)
# args = parser.parse_args()

parameters = Hyperparameters(input_dim=train_en[2], output_dim=train_hin[2], encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, cell_type=args.cell_type, bidirectional=args.bidirectional, hidden_size=args.hidden_neurons, learning_rate=args.learning_rate, optimizer=args.optimizer,dropout=args.dropout,embed_dim=args.embeded_dim)
encoder = EncoderRNN(parameters).to(device)
decoder = DecoderRNN(parameters).to(device)

# Train the models
train(encoder, decoder, args.epochs, parameters, train_data, valid_data, batch_size=BATCH_SIZE, teacher_forcing_ratio=0.5)
