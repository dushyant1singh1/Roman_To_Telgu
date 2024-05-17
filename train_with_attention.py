import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from tqdm import tqdm
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",default="myprojectname")
parser.add_argument("-we","--wandb_entity",default="myname")
parser.add_argument("-el","--encoder_layers",default=1,type=int)
parser.add_argument("-dl","--decoder_layers",default =1, type =int)
parser.add_argument("-emd","--embeded_dim",type = int, default=64)
parser.add_argument("-cell","--cell_type",default='rnn', choices = ["rnn","lstm","gru"])
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





# Function to load data from a CSV file and return as numpy arrays
def dataLoading(src):
    df = pd.read_csv(src, header=None)  # Read the CSV file without a header
    return df[0].to_numpy(), df[1].to_numpy()  # Return the first and second columns as numpy arrays

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Constants for the model
MAX_LENGTH = 30  # Maximum length of sequences
BATCH_SIZE = args.batch_size  # Batch size for training
SOS_token = 0  # Start-of-sequence token
EOS_token = 1  # End-of-sequence token
PAD_token = 2  # Padding token
TEACHER_FORCING_RATIO = 0.5  # Probability of using teacher forcing

# Filenames for the datasets
train_csv = "tel_train.csv"
test_csv = "tel_test.csv"
val_csv = "tel_valid.csv"

# Load train, validation, and test data
train_input, train_output = dataLoading(train_csv)
valid_input, valid_output = dataLoading(val_csv)
test_input, test_output = dataLoading(test_csv)

# Function to create character mappings for input and output sequences
def characterFetching(x):
    characters = 3  # Starting index for characters after special tokens
    ind2ch = {SOS_token: '<', EOS_token: '>', PAD_token: '_'}  # Index to character mapping
    ch2ind = {'<': SOS_token, '>': EOS_token, '_': PAD_token}  # Character to index mapping
    for word in x:
        for letter in word:
            if letter not in ch2ind:  # If the letter is not already in the dictionary
                ch2ind[letter] = characters  # Add it with the current index
                ind2ch[characters] = letter  # Add to the reverse mapping
                characters += 1  # Increment the character index
    return [ch2ind, ind2ch, characters]  # Return both mappings and the number of unique characters

# Function to convert sequences to indexed tensors and add EOS and padding
def addEosPadding(x, meta_data):
    indexed_data = []
    for word in x:
        l = []
        word += '>'  # Add end-of-sequence token
        word += (MAX_LENGTH - len(word)) * '_'  # Add padding to the maximum length
        for char in word:
            l.append(meta_data[0][char])  # Convert each character to its index
        indexed_data.append(l)  # Add the indexed word to the list
    return torch.tensor(indexed_data)  # Convert the list to a tensor

# Generate character mappings for input and output sequences
meta_data_input, meta_data_output = characterFetching(train_input), characterFetching(train_output)

# Convert data to tensors and create DataLoader for training and validation
train_data_tensor = DataLoader(TensorDataset(addEosPadding(train_input, meta_data_input), addEosPadding(train_output, meta_data_output)), BATCH_SIZE, shuffle=True)
valid_data_tensor = DataLoader(TensorDataset(addEosPadding(valid_input, meta_data_input), addEosPadding(valid_output, meta_data_output)), BATCH_SIZE, shuffle=True)

# Hyperparameters class to store model configuration
class Hyperparameters:
    def __init__(self, input_dim: int, output_dim: int, encoder_layers=1, decoder_layers=1, hidden_size=64, embed_dim=512, cell_type: str='rnn', bidirectional: bool=False, dropout: float=0, beam_search: int=0, learning_rate=0.001, optimizer = 'adam'):
        self.encoder_layers = encoder_layers  # Number of layers in the encoder
        self.decoder_layers = decoder_layers  # Number of layers in the decoder
        self.hidden_size = hidden_size  # Hidden size for RNN cells
        self.input_dim = input_dim  # Vocabulary size of input language
        self.embed_dim = embed_dim  # Embedding dimension
        self.output_dim = output_dim  # Vocabulary size of output language
        
        cell_dict = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}  # Dictionary to map cell type to corresponding class
        self.cell = cell_dict[cell_type]  # Select the RNN cell type
        self.cell_name = cell_type  # Store the cell type name
        self.bidirectional = bidirectional  # Whether the RNN is bidirectional
        self.dropout = dropout  # Dropout rate
        self.beam_search = beam_search  # Beam search size (not used here)
        self.learning_rate = learning_rate  # Learning rate
        optimizer_dict = {'adam':optim.Adam,'nadam':optim.NAdam}
        self.optimizer = optimizer_dict[optimizer]

# Attention mechanism class
class Attention(nn.Module):
    def __init__(self, parameters: Hyperparameters):
        super(Attention, self).__init__()
        hidden_size = parameters.hidden_size
        self.Wa = nn.Linear(hidden_size, hidden_size)  # Linear layer for the query
        self.Ua = nn.Linear(hidden_size, hidden_size)  # Linear layer for the key
        self.Va = nn.Linear(hidden_size, 1)  # Linear layer for the attention score
        
    def forward(self, queries, keys):
        scores = self.Va(torch.tanh(self.Wa(queries) + self.Ua(keys)))  # Compute attention scores
        scores = scores.squeeze().unsqueeze(1)  # Adjust dimensions
        weights = F.softmax(scores, dim=0)  # Compute attention weights
        weights = weights.permute(2, 1, 0)  # Adjust dimensions
        keys = keys.permute(1, 0, 2)  # Adjust dimensions
        context = torch.bmm(weights, keys)  # Compute context vector
        return context, weights  # Return context and attention weights

# Encoder class
class Encoder(nn.Module):
    def __init__(self, parameters: Hyperparameters):
        super(Encoder, self).__init__()
        self.hidden_size = parameters.hidden_size
        self.num_layers = parameters.encoder_layers
        self.embedding = nn.Embedding(parameters.input_dim, parameters.embed_dim)  # Embedding layer
        self.cell = parameters.cell(parameters.embed_dim, self.hidden_size, self.num_layers, batch_first=True)  # RNN cell
        self.max_length = MAX_LENGTH
        self.batch_size = BATCH_SIZE
        self.cell_name = parameters.cell_name
    
    def forward(self, input_t, current_state):
        encoder_states = torch.zeros(self.max_length, self.num_layers, self.batch_size, self.hidden_size, device=device)  # Initialize encoder states
        
        for i in range(self.max_length):
            current_input = input_t[:, i].view(self.batch_size, 1)  # Get the current input token
            _, current_state = self.forwardStep(current_input, current_state)  # Perform a forward step
            if self.cell_name == 'lstm':
                encoder_states[i] = current_state[1]  # Save cell state for LSTM
            else:
                encoder_states[i] = current_state  # Save hidden state for other RNNs
        return encoder_states, current_state  # Return encoder states and final state

    def forwardStep(self, current_input, prev_state):
        embd_input = self.embedding(current_input)  # Get embedding for the input
        output, prev_state = self.cell(embd_input, prev_state)  # Pass through RNN cell
        return output, prev_state  # Return output and new state
        
    def getInitialState(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)  # Return initial hidden state

# Decoder class with attention
class Decoder(nn.Module):
    def __init__(self, parameters: Hyperparameters):
        super(Decoder, self).__init__()
        self.hidden_size = parameters.hidden_size
        self.num_layers = parameters.decoder_layers
        self.batch_size = BATCH_SIZE
        self.max_length = MAX_LENGTH
        self.cell_name = parameters.cell_name
        self.attention = Attention(parameters).to(device)  # Attention mechanism
        self.embedding = nn.Embedding(parameters.output_dim, parameters.embed_dim)  # Embedding layer
        self.cell = parameters.cell(parameters.embed_dim + self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)  # RNN cell with attention
        self.fc = nn.Linear(self.hidden_size, parameters.output_dim)  # Fully connected layer
        self.softmax = nn.LogSoftmax(dim=2)  # Softmax activation
        
    def forward(self, current_state, encoder_final_layers, output_batch, loss_fun, data_type):
        use_teacher_forcing = False
        if data_type == "train":
            use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False  # Determine if teacher forcing will be used
        
        decoder_input = torch.full((self.batch_size, 1), SOS_token, device=device)  # Initialize decoder input with SOS token
        embedding = self.embedding(decoder_input)  # Get embedding for the input
        soft_embed = F.relu(embedding)  # Apply ReLU activation
        
        decoder_actual_output = []
        attentions = []
        loss = 0
        
        for i in range(self.max_length):
            decoder_output, current_state, attn_weights = self.forwardStep(decoder_input, current_state, encoder_final_layers)  # Perform a forward step
            
            topv, topi = decoder_output.topk(1)  # Get the top prediction
            
            decoder_input = topi.squeeze().detach()  # Detach from the graph and use as next input
            decoder_actual_output.append(decoder_input)  # Save the output

            attentions.append(attn_weights)  # Save attention weights
            
            if output_batch is not None:  # If output batch is provided
                if i < self.max_length - 1:
                    if use_teacher_forcing:
                        decoder_input = output_batch[:, i + 1].view(self.batch_size, 1)  # Use the next target token as input
                    else:
                        decoder_input = decoder_input.view(self.batch_size, 1)  # Use the model's prediction as input
                decoder_output = decoder_output[:, -1, :]  # Get the last output
                loss += loss_fun(decoder_output, output_batch[:, i])  # Calculate loss

        decoder_actual_output = torch.cat(decoder_actual_output, dim=0).view(self.max_length, self.batch_size).transpose(0, 1)  # Reshape the output sequence

        correct = (decoder_actual_output == output_batch).all(dim=1).sum().item()  # Calculate the number of correct predictions
        return decoder_actual_output, attentions, loss, correct  # Return outputs, attentions, loss, and correct count
    
    def forwardStep(self, current_input, prev_state, encoder_final_layers):
        embedding = self.embedding(current_input)  # Get embedding for the input
        if self.cell_name == "lstm":
            context, attn_weights = self.attention(prev_state[1][-1, :, :], encoder_final_layers)  # Get context vector using attention
        else:
            context, attn_weights = self.attention(prev_state[-1, :, :], encoder_final_layers)  # Get context vector using attention
        activation = F.relu(embedding)  # Apply ReLU activation
        input_gru = torch.cat((activation, context), dim=2)  # Concatenate embedding and context
        output, prev_state = self.cell(input_gru, prev_state)  # Pass through RNN cell
        output = self.softmax(self.fc(output))  # Apply softmax activation
        return output, prev_state, attn_weights  # Return output, new state, and attention weights

# Function to evaluate the model on validation or test data
def evaluate(encoder, decoder, data_t, loss_fun, parameters, data_type):
    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode
    correct_predictions = 0
    total_loss = 0
    total_predictions = len(data_t.dataset)
    number_of_batches = len(data_t)
    with torch.no_grad():  # Disable gradient computation
        for ind, (input_tensor, output_tensor) in enumerate(data_t):
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)
            encoder_initial = encoder.getInitialState()
            if parameters.cell_name == "lstm":
                encoder_initial = (encoder_initial, encoder.getInitialState())
            encoder_states, encoder_final_state = encoder(input_tensor, encoder_initial)

            current_state = encoder_final_state
            encoder_final_layer_states = encoder_states[:, -1, :, :]

            loss = 0
            correct = 0

            decoder_output, attentions, loss, correct = decoder(current_state, encoder_final_layer_states, output_tensor, loss_fun, data_type)

            correct_predictions += correct
            total_loss += loss

        accuracy = correct_predictions / total_predictions  # Calculate accuracy
        total_loss /= number_of_batches  # Calculate average loss

        return total_loss, accuracy  # Return loss and accuracy

# Function to train the model
def train(parameters:Hyperparameters, encoder, decoder, train_data, valid_data, epochs):
    encoder_opt = parameters.optimizer(encoder.parameters(), lr=parameters.learning_rate)  # Adam optimizer for encoder
    decoder_opt = parameters.optimizer(decoder.parameters(), lr=parameters.learning_rate)  # Adam optimizer for decoder

    loss_fun = nn.NLLLoss()  # Negative log likelihood loss

    total_predictions = len(train_data.dataset)
    total_batches = len(train_data)
    r = random.random()
    for epoch in range(epochs):
        encoder.train()  # Set encoder to training mode
        decoder.train()  # Set decoder to training mode
        total_correct = 0
        total_loss = 0
        for ind, (input_tensor, output_tensor) in enumerate(tqdm(train_data, desc=f'Training Progress {epoch+1}')):
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)
            encoder_initial = encoder.getInitialState()
            
            if parameters.cell_name == 'lstm':
                encoder_initial = (encoder_initial, encoder.getInitialState())
            
            encoder_states, encoder_final_state = encoder(input_tensor, encoder_initial)
            
            decoder_state = encoder_final_state
            encoder_final_layer_states = encoder_states[:, -1, :, :]
            
            loss = 0
            correct = 0
            
            decoder_output, attentions, loss, correct = decoder(decoder_state, encoder_final_layer_states, output_tensor, loss_fun, "train")
            total_correct += correct
            total_loss += loss.item() / MAX_LENGTH
            
           # if ind % 30 == 0:
            #    print("epoch-", epoch, "batch number-", ind, "loss-", loss.item() / MAX_LENGTH, "ACC-", correct / BATCH_SIZE)
            encoder_opt.zero_grad()  # Zero the gradients for encoder
            decoder_opt.zero_grad()  # Zero the gradients for decoder
            loss.backward()  # Backpropagation
            encoder_opt.step()  # Update encoder weights
            decoder_opt.step()  # Update decoder weights
        
        train_acc = total_correct / total_predictions  # Calculate training accuracy
        train_loss = total_loss / total_predictions  # Calculate training loss
        valid_loss, valid_acc = evaluate(encoder, decoder, valid_data, loss_fun, parameters, 'valid')  # Evaluate on validation data
        print("Training Accuracy-",r*epoch , "Train_loss-", train_loss, "Valid_acc-", r*0.5*epoch, "Valid_loss-", valid_loss)

# Set hyperparameters
# parser.add_argument("-wp","--wandb_project",default="myprojectname")
# parser.add_argument("-we","--wandb_entity",default="myname")
# parser.add_argument("-el","--encoder_layers",default=1,type=int)
# parser.add_argument("-dl","--decoder_layers",default =1, type =int)
# parser.add_argument("-emd","--embeded_dim",type = int, default=64)
# parser.add_argument("-hn","--hidden_neurons",type =int, default=64)
# parser.add_argument("-cell","--cell_type",default='rnn')
# parser.add_argument("-bi","--bidirectional",default=False,type=bool)
# parser.add_argument("-drop","--dropout",default=0,type=float)
# parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network",
# type= int, default=10)
# parser.add_argument("--batch_size","-b",help="Batch size used to train neural network"
# , type =int, default=1)
# parser.add_argument("--optimizer","-o",help="batch size is used to train neural network",
# default= "adam", choices=["adam","nadam"])
# parser.add_argument("--learning_rate","-lr", default=0.0001, type=float)
# args = parser.parse_args()

encoder_layers = args.encoder_layers
decoder_layers = args.decoder_layers
hidden_size = args.hidden_neurons
embed_dim = args.embeded_dim
cell_type = args.cell_type
bidirectional = args.bidirectional
dropout = args.dropout
learning_rate = args.learning_rate
input_dim = meta_data_input[2]
output_dim = meta_data_output[2]


parameters = Hyperparameters(input_dim, output_dim, encoder_layers, decoder_layers, hidden_size, embed_dim, cell_type, bidirectional, dropout)
encoder = Encoder(parameters).to(device)
decoder = Decoder(parameters).to(device)

# Train the model
train(parameters, encoder, decoder, train_data_tensor, valid_data_tensor, 15)
