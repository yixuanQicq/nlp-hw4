#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
# import tensorflow

import gensim
import transformers
import string

from typing import List


def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()


def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    result = []
    for syns in wn.synsets(lemma, pos=pos):
        for l in syns.lemmas():
            name = l.name()
            name1 = name
            if '_' in name1:
                name1 = name.replace('_', '')
            if name != lemma and name1 not in result:
                result.append(name)
    return set(result)


def smurf_predictor(context: Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context: Context) -> str:
    lemma = context.lemma
    pos = context.pos
    dict = {}
    for syns in wn.synsets(lemma, pos=pos):
        for l in syns.lemmas():
            name = l.name()
            if name != lemma:
                if name in dict:
                    dict[name] += l.count()
                else:
                    dict[name] = l.count()
    max_count_key = max(dict, key=dict.get)
    if '_' in max_count_key:
        max_count_key.replace('_', '')
    result = max_count_key
    return result


def wn_simple_lesk_predictor(context: Context) -> str:
    lemma = context.lemma
    pos = context.pos
    stop_words = set(stopwords.words('english'))
    max_overlap = 0
    max_syn = None
    context_join = set(context.left_context + context.right_context)

    lexemes = wn.lemmas(lemma, pos=pos)
    for syns in wn.synsets(lemma, pos=pos):
        curr = set()
        # definition
        definition = set(tokenize(syns.definition()))
        curr = curr.union(definition)
        # examples
        for example in syns.examples():
            ex = set(tokenize(example))
            curr = curr.union(ex)
        # all hypernyms
        for hypernyms in syns.hypernyms():
            # definition for hypernyms
            h = set(tokenize(hypernyms.definition()))
            curr.union(h)
            # examples for hypernyms
            for hex in hypernyms.examples():
                hex = set(tokenize(hex))
                curr.union(hex)
        # remove stop word
        curr = curr - stop_words
        # overlap
        overlap = 0
        for s in curr:
            if s in context_join:
                overlap += 1
        if overlap >= max_overlap:
            if lemma not in syns.lemma_names():
                max_overlap = overlap
                max_syn = syns

    if max_syn is None:
        max_count = 0
        max_lemma_name = None
        for syns in wn.synsets(lemma, pos=pos):
            for l in syns.lemmas():
                name = l.name()
                if name != lemma:
                    count = l.count()
                    if count >= max_count:
                        max_count = count
                        max_lemma_name = name
        if '_' in max_lemma_name:
            max_lemma_name.replace('_', '')
        result = max_lemma_name
        return result
    else:
        max_count = 0
        max_lemma_name = None
        for l in max_syn.lemmas():
            name = l.name()
            if name != lemma:
                count = l.count()
                if count >= max_count:
                    max_count = count
                    max_lemma_name = name
        if '_' in max_lemma_name:
            max_lemma_name.replace('_', '')
        result = max_lemma_name
        return result


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self, context: Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        dict = {}
        for candidate in candidates:
            try:
                dict[candidate] = self.model.similarity(context.lemma, candidate)
            except:
                continue
        result = max(dict, key=dict.get)
        return result


class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        
        return None  # replace for part 5


if __name__ == "__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        # prediction = smurf_predictor(context)
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        prediction = predictor.predict_nearest(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
