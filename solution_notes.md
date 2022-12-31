# ScienceBot
An 💬NLP chatbot **answering questions about science**.

## Pre-requisites
- Install poetry
```
pip install poetry
```

## Approach
When a question is asked, TF-IDF (“Term Frequency — Inverse Document Frequency”) is used to search a database of science facts and score the documents relating to the question. It is a technique to calculate the weight of each word in the question i.e. the importance of the word in the document and corpus. This algorithm is mostly using for the retrieval of information and text mining.

The [SciQ Dataset](https://allenai.org/data/sciq) is used as the document store (database). It contains 13,679 crowdsourced science exam questions about Physics, Chemistry and Biology, among others. 

When a question is asked, the bot computes the query vectors and then retrieves relevant context from document store if similarity score > threshold of 0.5. If score not up to threshold i.e., no relevant context found, the summary from the top Wikipedia page relating to the question is used as context.

A pretrained model: `DistilBERT` finetuned on the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset for question answering is used to extract answers from the context. This answer is then returned to the user.

## Steps
1. Create virtual environment and install dependencies
```
poetry install
```

2. Activate virtual environment
```
poetry shell
```

3. Download Spacy English language model.  
```
python -m spacy download en_core_web_sm
```

4. Optional - Run modelling.py to build document store. I've saved one to the [artifacts](artifacts/) directory.
```
python modelling.py
```

5. Start ScienceBot
```
python ignite.py
```
The bot should be running on port 5000

6. Test ScienceBot by sending a `POST` request to http://localhost:5000/app
