import os
import spacy
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

isExist = os.path.exists(r'artifacts')
if not isExist:
   os.makedirs(r'artifacts')

print('Loading dataset...')
sci_questions = load_dataset("sciq")
sci_questions = sci_questions.remove_columns(['distractor3', 'distractor1', 'distractor2'])
doc_context = sci_questions['train']['support'] + sci_questions['validation']['support'] + sci_questions['test']['support']

def lemmatize(text):
  result = []
  for doc in tqdm(nlp.pipe(text, batch_size=32, n_process=3,  disable=["parser", "ner"]), total=len(text)):
    lemmatized_text = " ".join(tok.lemma_ for tok in doc)
    result.append(lemmatized_text)
  return result

print('Lemmatizing...')
lemmed_context = lemmatize(doc_context)

print('Saving document store as feather')
df = pd.DataFrame(data={'context': doc_context, 'lemmas': lemmed_context})
df.to_feather(r"artifacts/doc_context.feather")

print('Vectorizing...')
vectorizer = TfidfVectorizer(
    stop_words='english', min_df=2, max_df=.8, ngram_range=(1,3))
tfidf = vectorizer.fit_transform(lemmed_context)

print('Saving vectorizer and vectors')
with open('artifacts/vectorizer.bin', 'wb') as f:
  pickle.dump((vectorizer, tfidf), f)