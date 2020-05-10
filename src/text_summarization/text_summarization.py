#!/usr/bin/env python
# coding: utf-8

# # TextRank

import os
import re
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

import nltk
import numpy as np
import pandas as pd

import settings
from utils import get_logger


class TextSummarizer:
    def __init__(self, model, stage, nof_output_sentences=3, debug=False):
        nltk.download('punkt')
        self.nof_output_sentences = nof_output_sentences
        self.stage = stage
        self.model = model
        self.debug = debug

        if self.debug:
            self.logger = get_logger('DEBUG')

    @staticmethod
    def load_data(stage: str) -> pd.DataFrame:

        file_path = os.path.join(settings.crawled_data_dir, f"{stage}.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            # stage_level = data[['text', 'stage_level']].groupby('stage_level').agg({'text': lambda text: ' '.join(text),
            # data = stage_level.reset_index(level=0)
            return data[data['text'].notnull()]

        else:
            # Crawl new data from Internet
            pass
            return pd.DataFrame()

    def _sentence_splitting(self, text: str):
        # Split text into sentences
        from nltk.tokenize import sent_tokenize
        text = re.sub('\.,', '. ', text)
        sentences = sent_tokenize(text)

        # sentences = [y for x in sentences for y in x] # flatten list
        if self.debug:
            self.logger.debug(f'First 3 sentences: {sentences[:3]}')
        return sentences

    @staticmethod
    def _glove_downloader():
        # ### Make sentences embeddings from GloVe
        glove_folder = os.path.join(settings.PROJECT_ROOT, 'src', 'text_summarization', 'GloVe_embeddings')
        file_path = os.path.join(glove_folder, 'glove.6B.zip')
        if not os.path.exists(file_path):
            urllib.request.urlretrieve('http://nlp.stanford.edu/data/glove.6B.zip', file_path)
            zip = zipfile.ZipFile(file_path)
            zip.extractall()
        return glove_folder
        # # GloVe Embeddings
        # get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
        # get_ipython().system('unzip glove*.zip')

    def _extract_word_embeddings_matrix(self):
        glove_folder = self._glove_downloader()
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

    @staticmethod
    def _text_preprocessing(sentences: list):
        # #### Text Preprocessing
        # Remove new-line character
        # sentences = [re.sub('\n+', ' ', sent) for sent in sentences]

        # Remove stopwords
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        stop_words = stopwords.words('english')

        def remove_stopwords(sen):
            sen_new = " ".join([i for i in sen if i not in stop_words])
            return sen_new

        cleaned_sentences = [remove_stopwords(r.split()) for r in sentences]
        return cleaned_sentences

    def _make_sentence_embeddings(self, word_embeddings: dict, cleaned_sentences: list):
        # #### Make a sentence vector from word embeddings avg.
        sentence_vectors = []
        for i in cleaned_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()))
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)

        assert len(cleaned_sentences) == len(sentence_vectors)

        # print('Sentence #0 has shape:', sentence_vectors[0].shape)
        return sentence_vectors

    def _create_similarity_matrix(self, cleaned_sentences: list):
        # ### Similarity Matrix Preparation
        # Similarity matrix is a zero matrix with dimension (n, n)
        # We will initialize this matrix with cosine similarity of the sentences
        sim_mat = np.zeros([len(cleaned_sentences), len(cleaned_sentences)])

        from sklearn.metrics.pairwise import cosine_similarity
        word_embeddings = self._extract_word_embeddings_matrix()
        sentence_embeddings = self._make_sentence_embeddings(word_embeddings, cleaned_sentences)
        for i in range(len(cleaned_sentences)):
            for j in range(len(cleaned_sentences)):
                if i != j:
                    sim_mat[i][j] = \
                        cosine_similarity(sentence_embeddings[i].reshape(1, 100),
                                          sentence_embeddings[j].reshape(1, 100))[0, 0]
        return sim_mat

    def text_rank_summarization(self, cleaned_sentences: list):
        # ### Applying PageRank algorithm
        #### Convert into graph

        # We need to convert the similarity matrix **sim_mat** into a graph.
        # The nodes of this graph will represent the sentences and the edges will represent the similarity scores between sentences.

        import networkx as nx

        sim_mat = self._create_similarity_matrix(cleaned_sentences)
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)

        # #### Summary Extraction
        # Extracting the top N sentences based on their rankings for summary generation

        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(cleaned_sentences)), reverse=True)

        # Extract some sentences as the summary
        top_k_sentences = [ranked_sentences[i][1] for i in range(self.nof_output_sentences)]
        summary = ' '.join(top_k_sentences)
        if self.debug:
            self.logger.debug(f'Top {self.nof_output_sentences}: {summary}')
        return summary

    def inference(self):

        data = self.load_data(stage)

        if self.model == 'TextRank':
            summarized_text = []

            for text in data['text']:
                sentences = self._sentence_splitting(text)
                cleaned_sentences = self._text_preprocessing(sentences)
                if len(cleaned_sentences) > self.nof_output_sentences:
                    top_k_sentences = self.text_rank_summarization(cleaned_sentences)
                else:
                    # No need to do summarizing
                    top_k_sentences = None
                summarized_text.append(top_k_sentences)

            data['summary'] = summarized_text

            return data
        else:
            pass


if __name__ == '__main__':
    # Choose a stage to use TextRank algorithm
    stage = 'stage 0'

    summarizer = TextSummarizer(stage=stage, model='TextRank', nof_output_sentences=3, debug=True)
    data_summarized = summarizer.inference()
    print(data_summarized)
