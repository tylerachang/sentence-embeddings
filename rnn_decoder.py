import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNDecoderModule(nn.Module):
    """LSTM RNN decoder NN module."""
    def __init__(self, vocab_size, embedding_size, n_hidden):
        super(RNNDecoderModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size,
                            num_layers=n_hidden)
        self.out = nn.Linear(embedding_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class RNNDecoder():
    """RNN decoder class. Wraps the RNNDecoderModule."""
    def __init__(self, vocab_size: int, embedding_size: int,
                 n_hidden: int, sos_token: int = 0, eos_token: int = 1,
                 max_output_length: int = 100) -> None:
        self.decoder = RNNDecoderModule(vocab_size, embedding_size, n_hidden)
        if torch.cuda.is_available(): self.decoder.cuda()
        self.n_hidden = n_hidden
        self.embedding_size = embedding_size
        self.SOS_token = sos_token
        self.EOS_token = eos_token
        self.max_output_length = max_output_length

    def _create_init_hidden(self, embedding):
        # TODO: maybe more of the cell/hidden states should be initialized
        # as the sentence embedding for more redundancy.
        # All hidden states except one start as zeros.
        decoder_h = torch.zeros(self.n_hidden-1, 1, self.embedding_size)
        decoder_h = torch.cat((embedding, decoder_h), 0)
        # All cell states start as zeros.
        decoder_c = torch.zeros(self.n_hidden, 1, self.embedding_size)
        if torch.cuda.is_available():
            decoder_c = decoder_c.cuda()
            decoder_h = decoder_h.cuda()
        return (decoder_h, decoder_c)

    def train(self, input_tensor, target_tensor, decoder_optimizer,
              criterion, teacher_forcing_ratio=0.5):
        """Train on a single sentence."""
        # TODO: Implement batching.
        decoder_optimizer.zero_grad()
        target_length = target_tensor.size(0) # Target sequence length.
        loss = 0
        decoder_input = torch.tensor([[self.SOS_token]], device=device)
        decoder_hidden = self._create_init_hidden(torch.reshape(input_tensor, (1, 1, -1)))
        if torch.cuda.is_available(): target_tensor = target_tensor.cuda()

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: feed the target as the next input.
            for i in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                loss += criterion(decoder_output, target_tensor[i])
                decoder_input = target_tensor[i]  # Teacher forcing.
        else:
            # Without teacher forcing: use prediction as the next input.
            for i in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                top_value, top_index = decoder_output.topk(1)
                decoder_input = top_index.squeeze().detach()  # Detach from graph.
                loss += criterion(decoder_output, target_tensor[i])
                if decoder_input.item() == self.EOS_token:
                    break

        loss.backward()
        decoder_optimizer.step()
        return loss.item() / target_length

    def train_iters(self, pairs, n_iters, print_every=1000, learning_rate=0.001):
        """Train for some number of iterations choosing randomly from the list of tensor pairs."""
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        print_loss_total = 0  # Reset every print_every.
        for iter in range(n_iters):
            # TODO: train through a random permutation of the examples instead of
            # selecting random examples.
            training_pair = random.choice(pairs)
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor, decoder_optimizer, criterion)
            print_loss_total += loss

            if iter % print_every == print_every-1:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('Examples: {0}\nAverage loss: {1}'.format(iter, print_loss_avg))

    def predict(self, input_tensor):
        with torch.no_grad():
            decoder_input = torch.tensor([[self.SOS_token]], device=device)
            decoder_hidden = self._create_init_hidden(torch.reshape(input_tensor, (1, 1, -1)))
            decoded_seq = []
            for i in range(self.max_output_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                top_value, top_index = decoder_output.topk(1)
                decoder_input = top_index.squeeze().detach()  # Detach from graph.
                if top_index.item() == self.EOS_token:
                    break
                else:
                    decoded_seq.append(top_index.item())
        return decoded_seq
