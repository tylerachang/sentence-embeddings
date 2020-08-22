import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from seq2seq.models import DecoderRNN, TopKDecoder
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNDecoder():
    """RNN decoder class. Wraps the IBM seq2seq decoder (using GRU or LSTM units)."""
    def __init__(self, vocab_size: int, embedding_size: int,
                 n_hidden: int, sos_token: int = 0, eos_token: int = 1, mask_token: int = 2,
                 max_output_length: int = 100, rnn_cell: str = 'lstm') -> None:
        self.decoder = DecoderRNN(vocab_size, max_output_length, embedding_size,
            n_layers=n_hidden, rnn_cell=rnn_cell, use_attention=False, bidirectional=False, eos_id=eos_token, sos_id=sos_token)
        if torch.cuda.is_available(): self.decoder.cuda()

        self.rnn_cell = rnn_cell
        self.n_hidden = n_hidden
        self.embedding_size = embedding_size
        self.SOS_token = sos_token
        self.EOS_token = eos_token
        self.mask_token = mask_token
        self.max_output_length = max_output_length
        token_weights = torch.ones(vocab_size)
        if torch.cuda.is_available(): token_weights=token_weights.cuda()
        self.loss = NLLLoss(weight=token_weights, mask=mask_token)
        self.optimizer = None

    def _create_init_hidden(self, embedding):
        # All hidden states start as the embedding.
        decoder_hidden = []
        for i in range(self.n_hidden):
            decoder_hidden.append(embedding)
        # num_layers x batch_size x embedding_size
        decoder_h = torch.cat(decoder_hidden, 0)
        return decoder_h

    def train(self, input_tensor, target_tensor, teacher_forcing_ratio=0.5):
        """Train for one batch."""
        decoder_outputs, decoder_hidden, ret_dict = self.decoder(
            inputs=target_tensor, encoder_hidden=input_tensor, teacher_forcing_ratio=teacher_forcing_ratio)
        # Nothing was generated. This number (10) was arbitrarily chosen.
        if len(decoder_outputs) == 0:
            return 10

        loss = self.loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_tensor.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_tensor[:, step + 1])
        self.decoder.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.get_loss()

    def train_iters(self, pairs, n_iters, batch_size=64, print_every=1000, learning_rate=0.0002, teacher_forcing_ratio=0.5):
        """Train for some number of iterations choosing randomly from the list of tensor pairs."""
        print("Initializing training.")
        if self.optimizer == None:
            adam = optim.Adam(self.decoder.parameters(), lr=learning_rate)
            self.optimizer =  Optimizer(adam, max_grad_norm=5)
        else:
            print("Using existing optimizer.")
        random.shuffle(pairs)
        if (len(pairs) < batch_size):
            print("Not enough examples for one batch.")
            return

        # Turn the pairs into big tensors.
        # TODO: instead of saving pairs, save tensors directly. Otherwise this operation takes too much space.
        # Input: num_layers x num_examples x embedding_size
        # Target: num_examples x max_output_length+1
        input_tensors = [ torch.reshape(i, (1, 1, -1)) for i, j in pairs ]
        input_tensor = torch.cat(input_tensors, 1)
        input_tensor = self._create_init_hidden(input_tensor)
        target_tensors = [ j for i, j in pairs ]
        targets = []
        for target in target_tensors:
            target_tensor = torch.reshape(target, (1,-1))
            if target_tensor.size(1) >= self.max_output_length:
                target_tensor = target_tensor[0][0:self.max_output_length]
                target_tensor = torch.reshape(target_tensor, (1,-1))
            else:
                pad = torch.zeros(1, self.max_output_length-target_tensor.size(1)).long()
                for i in range(self.max_output_length-target_tensor.size(1)):
                    pad[0][i] = self.mask_token
                target_tensor = torch.cat((target_tensor, pad), 1)
            # Add the start token.
            start_tensor = torch.zeros(1,1).long()
            start_tensor[0][0] = self.SOS_token
            target_tensor = torch.cat((start_tensor, target_tensor), 1)
            targets.append(target_tensor)
        target_tensor = torch.cat(targets, 0)

        if torch.cuda.is_available(): target_tensor = target_tensor.cuda()
        if torch.cuda.is_available(): input_tensor = input_tensor.cuda()

        print("Starting training.")
        print_loss_total = 0  # Reset every print_every.
        batch = 0
        for iter in range(n_iters):
            # Create the batch.
            if (batch+1)*batch_size > len(pairs):
                print("Finished an epoch!")
                batch = 0
            batch_input = input_tensor[:,batch*batch_size:(batch+1)*batch_size,:].contiguous()
            batch_target = target_tensor[batch*batch_size:(batch+1)*batch_size,:].contiguous()

            if self.rnn_cell == 'lstm':
                batch_input = (batch_input, batch_input)

            loss = self.train(batch_input, batch_target, teacher_forcing_ratio=teacher_forcing_ratio)
            print_loss_total += loss

            if iter % print_every == print_every-1:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('Steps: {0}\nAverage loss: {1}'.format(iter, print_loss_avg))
            batch += 1

    def predict(self, input_tensor, beam_size: int):
        if beam_size > 1:
            beam_decoder = TopKDecoder(self.decoder, beam_size)
        else:
            beam_decoder = self.decoder
        with torch.no_grad():
            decoder_hidden = self._create_init_hidden(torch.reshape(input_tensor, (1, 1, -1)))
            if torch.cuda.is_available(): decoder_hidden = decoder_hidden.cuda()
            if self.rnn_cell == 'lstm':
                decoder_hidden = (decoder_hidden, decoder_hidden)
            decoder_outputs, decoder_hidden, ret_dict = beam_decoder(
                inputs=None, encoder_hidden=decoder_hidden, teacher_forcing_ratio=0)
        output_sequence = []
        for item in ret_dict['sequence']:
            output_sequence.append(item[0].item())
        return output_sequence
