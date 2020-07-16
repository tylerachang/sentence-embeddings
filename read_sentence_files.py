"""
Reads sentences from files.
"""

from __future__ import unicode_literals

import os
import codecs
import argparse
from nltk.tokenize import sent_tokenize

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    return parser

def main(directory):
    outfile = codecs.open("output.txt", 'w', encoding='utf-8')
    sentence_count = 0
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if not 'txt' in fname:
                continue
            filename = os.path.join(root, fname)
            print("Writing " + filename)
            file = codecs.open(filename, 'rb', encoding='utf-8')
            text = ""
            for line in file:
                text = text + line
            sentences = sent_tokenize(text)
            for sentence in sentences:
                # Remove multiple whitespaces.
                words = sentence.split()
                if len(words) < 3:
                    continue # Remove extremely short sentences.
                cleaned_sentence = ' '.join(words).strip()
                # Remove underscores.
                cleaned_sentence = cleaned_sentence.replace("_", "")
                outfile.write(cleaned_sentence)
                outfile.write('\n')
                sentence_count += 1
    print('Done; final sentence count: {}'.format(sentence_count))


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.input_dir)
