#!/usr/bin/env python
# coding: utf-8

# # TextRank

# Automatic text summarization is the task of producing a concise and fluent summary while preserving key information
# content and overall meaning.
# 
# 1. Extractive Summarization
#  - Identifying the important sentences or phrases from the original text and extract only those from the text.
# 
# 2. Abstractive Summarization
#  - Generating new sentences from the original text
# 
# 
# 3. TextRank: extractive & unsupervised text summarizatoin
#  -  Concatenate text -> sentences -> sentence embeddings -> similarity matrix (between vectors) -> graph
import urllib
import zipfile

import numpy as np
import pandas as pd
import nltk
import settings

nltk.download('punkt')
import re
import os


def load_data(stage: str):

    file_path = os.path.join(settings.PROJECT_ROOT, 'data_crawled', f"{stage}.csv")
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path)
        # stage_level = data[['text', 'stage_level']].groupby('stage_level').agg({'text': lambda text: ' '.join(text),
        # data = stage_level.reset_index(level=0)
        return data['text'].dropna().values

    else:
        # Crawl new data from Internet
        data = None
        pass
        return data

# Split text into sentences
from nltk.tokenize import sent_tokenize

def sentence_spliting(text: str):
    sentences = sent_tokenize(text)

    # sentences = [y for x in sentences for y in x] # flatten list
    print('First 3 sentences:', sentences[:3])
    return sentences

# ### Make sentences embeddings from GloVe

def glove_downloader():
    glove_folder = os.path.join(settings.PROJECT_ROOT, 'text_summarization', 'GloVe_embeddings')
    file_path = os.path.join(glove_folder, 'glove.6B.zip')
    if not os.path.isfile(file_path):
        urllib.request.urlretrieve('http://nlp.stanford.edu/data/glove.6B.zip', file_path)
        zip = zipfile.ZipFile(file_path)
        zip.extractall()
    return glove_folder
    # # GloVe Embeddings
    # get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
    # get_ipython().system('unzip glove*.zip')


def extract_word_embeddings_matrix():
    glove_folder = glove_downloader()
    # Extract word vectors
    word_embeddings = {}
    f = open(os.path.join(glove_folder, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

# #### Text Preprocessing


def text_preprocessing(sentences: list):
    # Remove new-line character
    cleaned_sentences = [re.sub('\n+', ' ', sent) for sent in sentences]

    # Remove stopwords
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')


    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    cleaned_sentences = [remove_stopwords(r.split()) for r in cleaned_sentences]

    return cleaned_sentences



# #### Make sentence vectors from word embeddings

def make_sentence_embeddings(word_embeddings: dict, cleaned_sentences: list):

    sentence_vectors = []
    for i in cleaned_sentences:
      if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split()))
      else:
        v = np.zeros((100,))
      sentence_vectors.append(v)

    assert len(cleaned_sentences) == len(sentence_vectors)

        # print('Sentence #0 has shape:', sentence_vectors[0].shape)
    return sentence_vectors




# ### Similarity Matrix Preparation

def create_similarity_matrix(clean_sentences: list):

    # Similarity matrix is a zero matrix with dimension (n, n)
    # We will initialize this matrix with cosine similarity of the sentences
    sim_mat = np.zeros([len(clean_sentences), len(clean_sentences)])

    from sklearn.metrics.pairwise import cosine_similarity
    word_embeddings =  extract_word_embeddings_matrix()
    sentence_embeddings = make_sentence_embeddings(word_embeddings, cleaned_sentences)
    for i in range(len(clean_sentences)):
      for j in range(len(clean_sentences)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_embeddings[i].reshape(1,100), sentence_embeddings[j].reshape(1,100))[0,0]
    return sim_mat



# ### Applying PageRank algorithm
#### Convert into graph

def page_rank_summarization(cleaned_sentences: list, nof_output_sentencs: int):
    # We need to convert the similarity matrix **sim_mat** into a graph.
    # The nodes of this graph will represent the sentences and the edges will represent the similarity scores between sentences.

    import networkx as nx

    sim_mat = create_similarity_matrix(cleaned_sentences)
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)


    # #### Summary Extraction
    # Extracting the top N sentences based on their rankings for summary generation

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # Extract some sentences as the summary
    for i in range(nof_output_sentencs):
          print(ranked_sentences[i][1])
    return ranked_sentences

if __name__ == '__main__':
    # Choose a stage to use TextRank algorithm
    stage = 'stage 0'

    texts = load_data(stage)
    for text in texts:
        sentences = sentence_spliting(text)
        cleaned_sentences = text_preprocessing(sentences)
        page_rank_summarization(cleaned_sentences, nof_output_sentencs=3)