"""
Sample usage of the SentenceGenerator class.
"""

from sentence_generator import SentenceGenerator
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from nltk.corpus import webtext
from nltk.corpus import inaugural
import dill as pickle # Needed to pickle lambda functions.
import os.path

model_id = "initial"
train = True
use_pickled_pairs = True
pickled_pairs = "drive/My Drive/training_pairs-{}.pickle".format('initial')
pickled_vocab = "drive/My Drive/vocab-{}.pickle".format('initial')

def detok_sentences(tokenized_sentences: list) -> list:
    sentences = []
    for tok_sent in tokenized_sentences:
        sentences.append(' '.join(tok_sent).strip())
    return sentences

embed_model = SentenceTransformer('bert-base-nli-mean-tokens')
embedding_fn = lambda s: embed_model.encode([s])[0]

all_sentences = []
if train and not use_pickled_pairs:
    print("Loading sentences.")
    nltk.download('brown')
    brown_tokenized_sentences = brown.sents()
    brown_sentences = detok_sentences(brown_tokenized_sentences)
    nltk.download('gutenberg')
    nltk.download('punkt')
    gutenberg_tokenized_sentences = gutenberg.sents()
    gutenberg_sentences = detok_sentences(gutenberg_tokenized_sentences)
    nltk.download('reuters')
    reuters_tokenized_sentences = reuters.sents()
    reuters_sentences = detok_sentences(reuters_tokenized_sentences)
    nltk.download('webtext')
    webtext_tokenized_sentences = webtext.sents()
    webtext_sentences = detok_sentences(webtext_tokenized_sentences)
    nltk.download('inaugural')
    inaugural_tokenized_sentences = inaugural.sents()
    inaugural_sentences = detok_sentences(inaugural_tokenized_sentences)
    all_sentences = brown_sentences + gutenberg_sentences + reuters_sentences + webtext_sentences + inaugural_sentences

model_save_path = "drive/My Drive/sentence_generator-{}.pickle".format(model_id)
if os.path.isfile(model_save_path):
    print("Loading sentence generator.")
    sentence_generator = pickle.load(open(model_save_path, "rb"))
    # Flattening is different depending on the decoder used in the sentence generator.
    # sentence_generator._decoder.decoder.lstm.flatten_parameters() # VanillaRNNDecoder
    # sentence_generator._decoder.decoder.rnn.flatten_parameters() # DecoderRNN (RNNDecoder with beam_size == 1)
    sentence_generator._decoder.decoder.rnn.rnn.flatten_parameters() # TopKDecoder (RNNDecoder with beam size > 1)
    print("Loaded sentence generator.")
else:
    sentence_generator = SentenceGenerator(embedding_fn, id=model_id)

if train:
    # Note: pickled vocab is only used if a vocab does not already exist.
    if not use_pickled_pairs:
        pickled_pairs = ""
        pickled_vocab = ""
    sentence_generator.train_generator(all_sentences, 20000,
        pickled_pairs=pickled_pairs,
        pickled_vocab=pickled_vocab,
        batch_size=32, learning_rate=0.0002, beam_size=3, rnn_cell='lstm')
    pickle.dump(sentence_generator, open(model_save_path, "wb"))

# Test the model.

emb = sentence_generator.get_embedding("This is a test sentence.")
sent = sentence_generator.get_sentence(emb)
print(sent)

emb = sentence_generator.get_embedding("This is also a test.")
sent = sentence_generator.get_sentence(emb)
print(sent)
