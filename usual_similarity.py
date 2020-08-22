"""
Get the usual cosine similarity between two sentence embeddings. Outputs
percentiles of pairwise cosine similarity scores.
"""

from sentence_transformers import SentenceTransformer
import os.path
import codecs
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embed_model = SentenceTransformer('bert-base-nli-mean-tokens')
embedding_fn = lambda s: embed_model.encode([s.replace("@@ ", "").replace("@@", "")])[0]

print("Loading sentences.")
all_sentences = []
indices = []
file = codecs.open("gutenberg-bpe-shuffled.txt", 'rb', encoding='utf-8')
i = 0
for line in file:
    if i >= 1000:
        break
    all_sentences.append(line.strip())
    indices.append(i)
    i += 1

print("Getting embeddings.")
embeddings = []
for sentence in all_sentences:
    embeddings.append(embedding_fn(sentence))

print("Getting cosine similarities.")
iters = 50000
similarities = np.zeros(iters)
similar_pair_count = 0
for i in range(iters):
    index_pair = random.sample(indices, 2)
    index1, index2 = tuple(index_pair)
    sim = cosine_similarity([embeddings[index1]], [embeddings[index2]])
    if sim > 0.8:
        print("PAIR:")
        print(all_sentences[index1].replace("@@ ", "").replace("@@", ""))
        print(all_sentences[index2].replace("@@ ", "").replace("@@", ""))
        similar_pair_count += 1
    similarities[i] = sim
print("{} similar pairs.".format(similar_pair_count))

print("\nOutput:")
for i in range(1001):
    print('Percentile {0}: \t{1}'.format(i/10, np.percentile(similarities, i/10)))
