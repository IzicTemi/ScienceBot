FROM openfabric/openfabric-pyenv:0.1.9-3.8
RUN pip install -U poetry
RUN mkdir cognitive-assistant
WORKDIR /cognitive-assistant
COPY . .
RUN poetry install -vvv --no-dev
RUN python -m spacy download en_core_web_sm
EXPOSE 5000
CMD ["sh","start.sh"]