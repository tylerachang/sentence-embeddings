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
import codecs

model_id = "gutenberg-10shards"
train = True
use_pickled_shards = True
use_pickled_vocab = True
use_pickled_pairs = False
num_train_iters = 1 # iters means epochs if using pickled_shards, otherwise means steps.

pickled_pairs = "drive/My Drive/training_pairs-{}.pickle".format(model_id)
pickled_vocab = "drive/My Drive/vocab-{}.pickle".format("gutenberg")
# A list of pickled_pairs files.
pickled_shards = ["drive/My Drive/training_pairs-gutenberg-shard{}.pickle".format(i+1) for i in range(10)]

# To use input sentences (will only be used if pickled_shards and pickled_pairs are not used).
all_sentences = []
file = codecs.open("drive/My Drive/gutenberg-bpe-shuffled.txt", 'rb', encoding='utf-8')
for line in file:
    all_sentences.append(line.strip())
all_sentences = all_sentences[0:500000]

# ------------------------------------------------------------------------------
# Prepare default sentences if necessary.

# Used to detokenize the default sentences.
def detok_sentences(tokenized_sentences: list) -> list:
    sentences = []
    for tok_sent in tokenized_sentences:
        sentences.append(' '.join(tok_sent).strip())
    return sentences

# Only used if no sentences are provided.
def get_default_sentences() -> list:
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
    return brown_sentences + gutenberg_sentences + reuters_sentences + webtext_sentences + inaugural_sentences

# Load default sentences if none provided.
if train and len(all_sentences) == 0 and not use_pickled_pairs and len(pickled_shards) == 0:
    print("Loading sentences.")
    all_sentences = get_default_sentences()

# ------------------------------------------------------------------------------
# Prepare the actual model.
embed_model = SentenceTransformer('bert-base-nli-mean-tokens')
embedding_fn = lambda s: embed_model.encode([s.replace("@@ ", "").replace("@@", "")])[0]

# Load or create the model.
model_save_path = "drive/My Drive/sentence_generator-{}.pickle".format(model_id)
if os.path.isfile(model_save_path):
    print("Loading sentence generator.")
    sentence_generator = pickle.load(open(model_save_path, "rb"))
    # Flattening is different depending on the decoder used in the sentence generator.
    # sentence_generator._decoder.decoder.lstm.flatten_parameters() # VanillaRNNDecoder
    sentence_generator._decoder.decoder.rnn.flatten_parameters() # DecoderRNN
    print("Loaded sentence generator.")
else:
    sentence_generator = SentenceGenerator(embedding_fn, id=model_id)

if train:
    # Note: pickled vocab is only used if a vocab does not already exist (i.e.
    # when training a model from scratch).
    if not use_pickled_vocab:
        pickled_vocab = ""
    # Note: the training data used is either pickled_shards, pickled_pairs,
    # or all_sentences (in that priority).
    if not use_pickled_shards:
        pickled_shards = []
    if not use_pickled_pairs:
        pickled_pairs = ""
    sentence_generator.train_generator(all_sentences, num_train_iters,
        pickled_pairs=pickled_pairs,
        pickled_shards=pickled_shards,
        pickled_vocab=pickled_vocab,
        vocab_size=30000,
        max_output_length=50,
        teacher_forcing_ratio=0.5,
        batch_size=64, learning_rate=0.0002, rnn_cell='lstm')
    pickle.dump(sentence_generator, open(model_save_path, "wb"))

# Test the model.
emb = sentence_generator.get_embedding("This is a test sentence.")
sent = sentence_generator.get_sentence(emb)
print(sent)

emb = sentence_generator.get_embedding("This is also a test.")
sent = sentence_generator.get_sentence(emb)
print(sent)
