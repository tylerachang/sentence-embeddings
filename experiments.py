"""
Experiments using the SentenceGenerator class.
"""

from sentence_generator import SentenceGenerator
from sentence_transformers import SentenceTransformer
import dill as pickle # Needed to pickle lambda functions.
import os.path
import codecs
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 5,000,000 training examples for 3 epochs.
model_id = "gutenberg-10shards-3epochs"

# Necessary because the sentence_generator calls this external function.
embed_model = SentenceTransformer('drive/My Drive/sentence_embeddings_data/bert-base-nli-mean-tokens')
embedding_fn = lambda s: embed_model.encode([s.replace("@@ ", "").replace("@@", "")])[0]

# Load the generator.
model_save_path = "drive/My Drive/sentence_embeddings_data/sentence_generator-{}.pickle".format(model_id)
if os.path.isfile(model_save_path):
    print("Loading sentence generator.")
    sentence_generator = pickle.load(open(model_save_path, "rb"))
    # Flattening is different depending on the decoder used in the sentence generator.
    # sentence_generator._decoder.decoder.lstm.flatten_parameters() # VanillaRNNDecoder
    sentence_generator._decoder.decoder.rnn.flatten_parameters() # DecoderRNN
    print("Loaded sentence generator.")
else:
    print("No model found. Let the errors begin!")

# ------------------------------------------------------------------------------
# Some sample functions.

embeddings = []
# Process and print embeddings (do this after embeddings has been filled by another function).
def process_embeddings():
    global embeddings
    for i in range(len(embeddings)):
        sent = sentence_generator.get_sentence(embeddings[i], min_num_noise_samples=10, max_num_noise_samples=200, threshold=0.90, variance=0.02, beam_size=10)
        new_embedding = embedding_fn(sent)
        sim = cosine_similarity([embeddings[i]], [new_embedding])
        print("Sentence {0}: {1}\nSimilarity: {2}".format(i, sent, sim))

# Line from one representation to another.
def embedding_line(start_sent: str, end_sent: str, num_steps: int):
    global embeddings
    start_emb = embedding_fn(start_sent)
    end_emb = embedding_fn(end_sent)
    diff_emb = end_emb - start_emb
    embeddings = []
    for i in range(num_steps+1):
        embeddings.append(start_emb + i*diff_emb/num_steps)

# Apply a difference vector to a new start embedding.
def diff_embedding(start_sent: str, end_sent: str, new_start_sent: str):
    global embeddings
    start_emb = embedding_fn(start_sent)
    end_emb = embedding_fn(end_sent)
    new_start_emb = embedding_fn(new_start_sent)
    new_end_emb = new_start_emb + end_emb - start_emb
    embeddings = []
    embeddings.append(start_emb)
    embeddings.append(end_emb)
    embeddings.append(new_start_emb)
    embeddings.append(new_end_emb)

# Scale an embedding.
# Uses 0 as default start magnitude. End magnitudes can vary, but note that
# usual vector lengths change with different dimensionality. E.g. unit vector
# has length sqrt(n_dim).
def scale_embedding(sent: str, end_magnitude: float = 25, num_steps: int = 20):
    global embeddings
    emb = embedding_fn(sent)
    embeddings = []
    for i in range(num_steps+1):
        embeddings.append(emb*end_magnitude*i/num_steps)

# ------------------------------------------------------------------------------
# Run experiments.
print('\n')

# Single sentence.
embeddings.append(embedding_fn("This is not a cat."))
process_embeddings()
print('\n')

embedding_line("This is a dog.", "This is not a cat.", 5)
process_embeddings()
print('\n')

diff_embedding("The dog ate the bone.", "The cat ate the bone.", "I saw the dog.")
process_embeddings()
print('\n')

scale_embedding("The dog ate the bone.", end_magnitude=25, num_steps=20)
process_embeddings()
print('\n')

# Random sentences.
noise_samples = np.random.multivariate_normal(np.zeros(768), 0.3*np.identity(768), 30)
noise_samples = np.float32(noise_samples)
embeddings = []
for i in range(30):
    embeddings.append(noise_samples[i])
process_embeddings()
print('\n')
