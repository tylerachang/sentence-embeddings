"""
Generate a sample dataset of sentences.
"""

import codecs
import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from nltk.corpus import webtext
from nltk.corpus import inaugural

def detok_sentences(tokenized_sentences: list) -> list:
    sentences = []
    for tok_sent in tokenized_sentences:
        sentences.append(' '.join(tok_sent).strip())
    return sentences

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

outfile = codecs.open('output.txt', 'w')
for sentence in all_sentences:
    cleaned_sentence = sentence.replace(" ' s ", "'s ")
    cleaned_sentence = cleaned_sentence.replace("n ' t ", "n't ")
    cleaned_sentence = cleaned_sentence.replace(" ,", ",")
    cleaned_sentence = cleaned_sentence.replace(" .", ".")
    outfile.write('{}\n'.format(cleaned_sentence))
