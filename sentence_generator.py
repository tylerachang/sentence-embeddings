import torch
import numpy as np
from collections import Counter
from rnn_decoder import RNNDecoder
from vanilla_rnn_decoder import VanillaRNNDecoder
import pickle

class SentenceGenerator:
    """
    Class to handle sentence embeddings (sentence <-> embedding).
    Trains an RNN to generate sentences from embeddings.
    """
    def __init__(self, embedding_fn,
                tokenize_fn = lambda s: s.strip().split(' '),
                detokenize_fn = lambda toks: ' '.join(toks).strip(),
                id: str = ""
                ) -> None:
        self._embedding_fn = embedding_fn
        self._tokenize_fn = tokenize_fn
        self._detokenize_fn = detokenize_fn
        self._id = id

        # These variables are instantiated upon RNN training.
        self._decoder = None
        self._token2index = {}
        self._index2token = {}
        self._n_tokens = 0
        self._embedding_size = 0

    def get_embedding(self, sentence: str) -> torch.FloatTensor:
        return self._embedding_fn(sentence)

    def get_sentence(self, embedding: torch.FloatTensor) -> str:
        decoded_seq = self._decoder.predict(torch.from_numpy(embedding))
        tokenized_sentence = []
        for item in decoded_seq:
            tokenized_sentence.append(self._index2token[item])
        return self._detokenize_fn(tokenized_sentence)

    def train_generator(self, training_sentences: list, iters: int,
                        vocab_size: int = 15000, pickled_pairs: str = "",
                        pickled_vocab: str = "", batch_size: int = 64,
                        learning_rate: float = 0.0002, n_layers: int = 4,
                        max_output_length: int = 20, print_every: int = 1000,
                        beam_size: int = 1, rnn_cell: str = 'lstm') -> None:
        """
        Trains an RNN decoder. Overrides the existing decoder.
        Either provide a list of training sentences (strings), or the pickled
        vocab and pickled training pairs.
        """
        if pickled_pairs == "" and len(training_sentences) == 0:
            print("No training sentences provided.")
            return
        if iters == 0:
            return

        # Load the vocabulary only if there is no existing vocab.
        if self._n_tokens == 0:
            if pickled_vocab == "":
                print("Creating vocabulary.")
                token_counts = {}
                tokenized_sentences = []
                for sentence in training_sentences:
                    tokenized_sentence = self._tokenize_fn(sentence)
                    tokenized_sentences.append(tokenized_sentence)
                    for token in tokenized_sentence:
                        if token not in token_counts:
                            token_counts[token] = 1
                        else:
                            token_counts[token] += 1
                # Find the most common tokens.
                token_counts = Counter(token_counts)
                token_counts = token_counts.most_common(vocab_size)
                # Create the dictionaries.
                self._token2index = {}
                self._index2token = {0: "<SOS>", 1: "<EOS>", 2: "<MASK>", 3: "<UNK>"}
                self._n_tokens = 4
                for token, count in token_counts:
                    self._token2index[token] = self._n_tokens
                    self._index2token[self._n_tokens] = token
                    self._n_tokens += 1
                vocab_to_pickle = tuple([self._token2index, self._index2token, self._n_tokens])
                pickle.dump(vocab_to_pickle, open("drive/My Drive/vocab-{}.pickle".format(self._id), "wb"))
            else:
                print("Loading pickled vocabulary.")
                self._token2index, self._index2token, self._n_tokens = pickle.load(open(pickled_vocab, "rb"))

        # Create the training pairs.
        if pickled_pairs == "":
            print("Creating training pairs.")
            training_pairs = []
            for sent_i in range(len(training_sentences)):
                # Each target tensor will have an end token, but no start token.
                tokenized_sentence = tokenized_sentences[sent_i]
                target_tensor = torch.zeros(len(tokenized_sentence)+1, 1).long()
                for tok_i in range(len(tokenized_sentence)):
                    if tokenized_sentence[tok_i] in self._token2index:
                        target_tensor[tok_i][0] = self._token2index[tokenized_sentence[tok_i]]
                    else:
                        target_tensor[tok_i][0] = 3 # Unknown token.
                target_tensor[-1][0] = 1 # End of sentence token.
                input_tensor = torch.from_numpy(self._embedding_fn(training_sentences[sent_i]))
                training_pairs.append(tuple([input_tensor, target_tensor]))
            pickle.dump(training_pairs, open("drive/My Drive/training_pairs-{}.pickle".format(self._id), "wb"))
        else:
            print("Loading pickled pairs.")
            training_pairs = pickle.load(open(pickled_pairs, "rb"))

        if self._decoder != None:
            print("Resuming training.")
            self._decoder.train_iters(training_pairs, iters, batch_size=batch_size, print_every=print_every)
            return

        # Train the RNN.
        print("Training RNN.")
        self._embedding_size = training_pairs[0][0].nelement()
        self._decoder = RNNDecoder(self._n_tokens, self._embedding_size, n_layers, max_output_length=max_output_length, beam_size=beam_size, rnn_cell=rnn_cell)
        self._decoder.train_iters(training_pairs, iters, batch_size=batch_size, print_every=print_every, learning_rate=learning_rate)
