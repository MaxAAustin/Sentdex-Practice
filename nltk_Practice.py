# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 00:33:16 2017

@author: maaus
"""

from nltk.tokenize import sent_tokenize, word_tokenize

example_text = 'Hello there, how are you doing today? The weather is great and Python is awesome. The sky is blue. You shouldn\'t eat cardboard'

print(sent_tokenize(example_text))
print(word_tokenize(example_text))