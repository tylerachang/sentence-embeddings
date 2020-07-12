import torch
import numpy as np
from collections import Counter
from rnn_decoder import RNNDecoder

class SentenceGenerator:
    """
    Class to handle sentence embeddings (sentence <-> embedding).
    Trains an LSTM RNN to generate sentences from embeddings.
    """
    def __init__(self, embedding_fn,
                tokenize_fn = lambda s: s.strip().split(' '),
                detokenize_fn = lambda toks: ' '.join(toks).strip()
                ) -> None:
        self._embedding_fn = embedding_fn
        self._tokenize_fn = tokenize_fn
        self._detokenize_fn = detokenize_fn

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
                        vocab_size: int = 10000) -> None:
        """Trains an RNN decoder. Overrides the existing decoder."""
        if len(training_sentences) == 0 or iters == 0:
            return

        # Load the vocabulary.
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
        self._index2token = {0: "<SOS>", 1: "<EOS>", 2: "<UNK>"}
        self._n_tokens = 3
        for token, count in token_counts:
            self._token2index[token] = self._n_tokens
            self._index2token[self._n_tokens] = token
            self._n_tokens += 1

        # Create the training pairs.
        training_pairs = []
        for sent_i in range(len(training_sentences)):
            tokenized_sentence = tokenized_sentences[sent_i]
            target_tensor = torch.zeros(len(tokenized_sentence)+1, 1).long()
            for tok_i in range(len(tokenized_sentence)):
                if tokenized_sentence[tok_i] in self._token2index:
                    target_tensor[tok_i][0] = self._token2index[tokenized_sentence[tok_i]]
                else:
                    target_tensor[tok_i][0] = 2 # Unknown token.
            target_tensor[-1][0] = 1 # End of sentence token.
            input_tensor = torch.from_numpy(self._embedding_fn(training_sentences[sent_i]))
            training_pairs.append(tuple([input_tensor, target_tensor]))

        # Train the RNN.
        self._embedding_size = training_pairs[0][0].nelement()
        self._decoder = RNNDecoder(self._n_tokens, self._embedding_size, 4)
        self._decoder.train_iters(training_pairs, iters, print_every=1000, learning_rate=0.001)
