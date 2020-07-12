"""
Sample usage of the SentenceGenerator class.
"""

from sentence_generator import SentenceGenerator
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

embedding_fn = lambda s: model.encode([s])[0]

sentence_generator = SentenceGenerator(embedding_fn)
sentence_generator.train_generator(["This is a test sentence.", "This is also a test."], 10000)

emb = sentence_generator.get_embedding("This is a test sentence.")
sent = sentence_generator.get_sentence(emb)
print(sent)

emb = sentence_generator.get_embedding("This is also a test.")
sent = sentence_generator.get_sentence(emb)
print(sent)
