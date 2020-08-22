import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from rnn_decoder import RNNDecoder
from vanilla_rnn_decoder import VanillaRNNDecoder
import pickle
import time

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

    # Gets a sentence for the embedding by sampling nearby embeddings, outputting
    # the sample with maximum similarity between the target embedding and
    # Embedding(Sentence(sample)).
    def get_sentence(self, embedding: torch.FloatTensor, min_num_noise_samples: int = 10,
                     max_num_noise_samples: int = 100, threshold: float = 0.90,
                     variance: float = 0.02, beam_size: int = 10) -> str:
        noise_samples = np.random.multivariate_normal(np.zeros(self._embedding_size), variance*np.identity(self._embedding_size), max_num_noise_samples)
        noise_samples = np.float32(noise_samples)
        max_sim = -1
        best_sent = ""
        for i in range(max_num_noise_samples):
            noisy_emb = embedding + noise_samples[i]
            sample_sent = self.get_sentence_no_noise(noisy_emb, beam_size = beam_size)
            sample_emb = self.get_embedding(sample_sent)
            sample_sim = cosine_similarity([embedding], [sample_emb])[0].item()
            if sample_sim > max_sim:
                max_sim = sample_sim
                best_sent = sample_sent
            if max_sim > threshold and i >= min_num_noise_samples:
                return best_sent
        return best_sent

    # Gets the sentence decoded from a given embedding.
    def get_sentence_no_noise(self, embedding: torch.FloatTensor, beam_size: int) -> str:
        decoded_seq = self._decoder.predict(torch.from_numpy(embedding), beam_size=beam_size)
        tokenized_sentence = []
        for item in decoded_seq:
            if item == 1: # EOS token.
                break
            tokenized_sentence.append(self._index2token[item])
        detokenized_sentence = self._detokenize_fn(tokenized_sentence)
        # In case OpenNMT BPE detokenization is not included in the detokenize function.
        return detokenized_sentence.replace("@@ ", "").replace("@@", "")

    def train_generator(self, training_sentences: list, iters: int,
                        vocab_size: int = 30000, pickled_pairs: str = "",
                        pickled_shards: list = [],
                        pickled_vocab: str = "", batch_size: int = 64,
                        learning_rate: float = 0.0002, n_layers: int = 4,
                        max_output_length: int = 100, print_every: int = 1000,
                        teacher_forcing_ratio: float = 0.5,
                        rnn_cell: str = 'lstm') -> None:
        """
        Trains an RNN decoder. Overrides the existing decoder.
        Either provide a list of training sentences (strings), or the pickled
        vocab and pickled training pairs.
        """
        if len(pickled_shards) == 0 and pickled_pairs == "" and len(training_sentences) == 0:
            print("No training sentences provided.")
            return
        if iters == 0:
            return

        # Load the vocabulary only if there is no existing vocab.
        if self._n_tokens == 0:
            if pickled_vocab == "":
                print("Creating vocabulary.")
                if len(training_sentences) == 0:
                    print("Cannot create vocabulary without input sentences.")
                    return
                token_counts = {}
                for sentence in training_sentences:
                    tokenized_sentence = self._tokenize_fn(sentence)
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

        # Special case using shards.
        if len(pickled_shards) > 0:
            if self._decoder == None:
                print("Instantiating new RNN.")
                training_pairs = pickle.load(open(pickled_shards[0], "rb"))
                self._embedding_size = training_pairs[0][0].nelement()
                self._decoder = RNNDecoder(self._n_tokens, self._embedding_size, n_layers, max_output_length=max_output_length, rnn_cell=rnn_cell)
            else:
                print("Resuming training.")
            for i in range(iters): # iters now means the number of epochs through the training data.
                print("Starting epoch {}".format(i))
                for pickled_pairs in pickled_shards:
                    training_pairs = None
                    time.sleep(10) # Wait 10 seconds, hope garbage collection is done...
                    print("Loading shard: {}".format(pickled_pairs))
                    training_pairs = pickle.load(open(pickled_pairs, "rb"))
                    print("Training on shard.")
                    num_steps = len(training_pairs)//batch_size
                    self._decoder.train_iters(training_pairs, num_steps, batch_size=batch_size, print_every=print_every)
                    print("Finished shard.")
            print("Finished {} epochs!".format(iters))
            return

        # Otherwise, use pickled pairs or provided sentences.
        # Create the training pairs.
        if pickled_pairs == "": # Only use the provided sentences if no pickled pairs.
            print("Creating training pairs.")
            training_pairs = []
            for sent_i in range(len(training_sentences)):
                # Each target tensor will have an end token, but no start token.
                tokenized_sentence =  self._tokenize_fn(training_sentences[sent_i])
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
        self._decoder = RNNDecoder(self._n_tokens, self._embedding_size, n_layers, max_output_length=max_output_length, rnn_cell=rnn_cell)
        self._decoder.train_iters(training_pairs, iters, batch_size=batch_size, print_every=print_every, learning_rate=learning_rate, teacher_forcing_ratio=teacher_forcing_ratio)
