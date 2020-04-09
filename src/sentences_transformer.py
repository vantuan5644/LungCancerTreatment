from sentence_transformers import SentenceTransformer


class TransformerSentencesEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.embeddings_dim = self.model.get_sentence_embedding_dimension()

    def sentences_encode(self, sentences):
        return self.model.encode(sentences)


if __name__ == "__main__":
    BERT = TransformerSentencesEmbedding()

    sentence = ['If you only have one small tumor in your lung and there is no evidence of cancer in lymph nodes or elsewhere, your doctors might recommend surgery to remove the tumor and the nearby lymph nodes.']
    embeddings = BERT.sentences_encode(sentence)
    print(embeddings[0].shape)

    pass