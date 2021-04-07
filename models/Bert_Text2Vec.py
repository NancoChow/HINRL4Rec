import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.models import Model
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel
import sys


class BERT_Text2Vec(object):

    def __init__(self, TOKEN_EMB_DIM=768, SENT_EMB_DIM=768, DOC_EMB_DIM=128):

        self.TOKEN_EMB_DIM = TOKEN_EMB_DIM
        self.SENT_EMB_DIM = SENT_EMB_DIM
        self.DOC_EMB_DIM = DOC_EMB_DIM

        # Load pre-trained model tokenizer (vocabulary)
        self.bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load pre-trained model (weights)
        self.bertModel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.bertModel.eval()

        # Initializing the Bi-LSTM encoders
        self.sent_BiLSTM_encoder = self.init_BiLSTM_encoder(self.SENT_EMB_DIM, self.SENT_EMB_DIM)
        self.doc_BiLSTM_encoder = self.init_BiLSTM_encoder(self.SENT_EMB_DIM, self.DOC_EMB_DIM)

    def process(self, document):
        sentences = sent_tokenize(document)
        sentence_embeddings = []
        for sentence in sentences:
            token_embeddings = self.learn_word_representation(sentence)
            sentence_embedding = self.learn_sent_representation(token_embeddings)
            sentence_embeddings.append(sentence_embedding)
        doc_embedding = self.learn_doc_representation(sentence_embeddings)
        return doc_embedding

    def init_BiLSTM_encoder(self, INPUT_DIM, OUTPUT_DIM):
        encoder_inputs = Input(shape=(1, INPUT_DIM))
        lstm = LSTM(OUTPUT_DIM, return_sequences=True, return_state=True)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = \
            Bidirectional(lstm, input_shape=(1, INPUT_DIM))(encoder_inputs)

        state_c = Concatenate()([forward_c, backward_c])
        state_h = Concatenate()([forward_h, backward_h])

        state_h = tf.reshape(state_h, [1, OUTPUT_DIM * 2, 1])

        max_pool = MaxPool1D(pool_size=2, strides=2, padding='valid')
        state_h = max_pool(state_h)
        encoder = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_c, state_h])
        return encoder

    def learn_doc_representation(self, sentence_embeddings):
        encoder_outputs, state_c, state_h = None, None, None
        for sentence_embedding in sentence_embeddings:
            sentence_embedding = sentence_embedding.reshape(-1, 1, self.SENT_EMB_DIM)
            encoder_outputs, state_c, state_h = self.doc_BiLSTM_encoder.predict(sentence_embedding, verbose=0)
        state_h = state_h.reshape(1, self.DOC_EMB_DIM)
        return state_h

    def learn_sent_representation(self, token_vecs):
        encoder_outputs, state_c, state_h = None, None, None
        for token_vec in token_vecs:
            token_vec = token_vec.reshape(-1, 1, self.TOKEN_EMB_DIM)
            encoder_outputs, state_c, state_h = self.sent_BiLSTM_encoder.predict(token_vec, verbose=0)
        return state_h

    def learn_word_representation(self, sentence):
        # Split the sentence into tokens
        indexed_tokens = self.bertTokenizer.encode(sentence, add_special_tokens=True, truncation=True)

        # Mark each of token as belonging to sentence "1".
        segments_ids = [1] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.bertModel(tokens_tensor, segments_tensors)
            # `hidden_states` has shape [13 x 1 x number_token_in_sentence x 768]
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_sum_vecs = []
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor
            # Sum the vectors from the all BERT's layers.
            token_sum_vec = torch.sum(token[-12:], dim=0)
            token_sum_vecs.append(token_sum_vec.numpy().reshape(self.TOKEN_EMB_DIM, 1))

        return token_sum_vecs

    def learn_BERT_sent_representation(self, sentence):
        # Split the sentence into tokens
        indexed_tokens = self.bertTokenizer.encode(sentence, add_special_tokens=True, truncation=True)

        # Mark each of token as belonging to sentence "1".
        segments_ids = [1] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.bertModel(tokens_tensor, segments_tensors)
            # `hidden_states` has shape [13 x 1 x number_token_in_sentence x 768]
            hidden_states = outputs[2]

        # `token_vecs` is a tensor with shape [number_token_in_sentence x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)

        return sentence_embedding


def main():
    sample_documents = [
        'Data mining is a process used by companies to turn raw data into useful information. By using software to look for patterns in large batches of data, businesses can learn more about their customers to develop more effective marketing strategies, increase sales and decrease costs. Data mining depends on effective data collection, warehousing, and computer processing.',
        'The process of digging through data to discover hidden connections and predict future trends has a long history.'
    ]
    TOKEN_EMB_DIM = 768
    SENT_EMB_DIM = 768
    DOC_EMB_DIM = 100
    model = BERT_Text2Vec(TOKEN_EMB_DIM, SENT_EMB_DIM, DOC_EMB_DIM)
    for document in sample_documents:
        doc_embedding = model.process(document)
        print(doc_embedding)


if __name__ == "__main__":
    sys.exit(main())
