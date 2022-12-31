import os
import warnings
import spacy
import pickle
import numpy as np
import pandas as pd
from mediawiki import MediaWiki

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

print('Loading tokenizer and model')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

wiki = MediaWiki()
nlp = spacy.load("en_core_web_sm")

print('Loading vectorizer and vectors')
with open('artifacts/vectorizer.bin', 'rb') as f_in:
    vectorizer, tfidf = pickle.load(f_in) 

print('Loading document store')
df = pd.read_feather(r'artifacts/doc_context.feather')

doc_context = df['context']

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        context = retrieve_context(text)
        response = extract_answer(text, context)
        if len(response) == 0:
            response = "Please reframe your question."
        output.append(response.capitalize())

    return SimpleText(dict(text=output))

def lemmatize(text):
  result = []
  for doc in (nlp.pipe(text, batch_size=32, disable=["parser", "ner"])):
    lemmatized_text = " ".join(tok.lemma_ for tok in doc)
    result.append(lemmatized_text)
  return result

def retrieve_context(question):
  query = vectorizer.transform(lemmatize([question]))
  threshold = 0.5
  scores = (tfidf * query.T).toarray()
  max_score = np.flip(np.sort(scores, axis=0))[0, 0]
  if max_score >= threshold:
    result = np.flip(np.argsort(scores, axis=0))[0, 0]
    context = doc_context[result]
  else:
    results = wiki.search(question)
    try:
      page = wiki.page(results[0])
      context = page.summary
    except mediawiki.exceptions.PageError:
      result = np.flip(np.argsort(scores, axis=0))[0, 0]
      context = doc_context[result]
  return context

def extract_answer(question, context):
  inputs = tokenizer(question, context, return_tensors="pt", truncation="only_second")
  with torch.no_grad():
      outputs = model(**inputs)

  answer_start_index = torch.argmax(outputs.start_logits)
  answer_end_index = torch.argmax(outputs.end_logits)

  predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
  answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
  return answer