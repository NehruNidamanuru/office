import os
import sys
import logging
from flask import Flask, request, jsonify
from flask import Flask, render_template, request
from werkzeug import secure_filename
import re
import flask
import os
from sklearn.model_selection import train_test_split

from keras.layers import concatenate
from sklearn.metrics import mean_squared_error
import nltk
import pandas as pd
import numpy as np
from nltk import word_tokenize          
from nltk.corpus import stopwords
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, Bidirectional,TimeDistributed
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model
from sklearn.metrics import roc_auc_score
import pickle
import tensorflow as tf
#graph = tf.get_default_graph()



# Define the app
app = Flask(__name__)


# Load the model
##Attention Layer

def dot_product(x, kernel):
    
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):    

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

import keras
global model_n1
global graph
graph = tf.get_default_graph()

model_n1 = keras.models.load_model('my_GRU_model.h5', custom_objects={'AttentionWithContext':AttentionWithContext()})

model_n1.compile(Adam(lr=0.01), 'categorical_crossentropy', metrics=['accuracy'])
model_n1.load_weights("model_doc_tagging_attention_GRU_glove_vec_100d_new.h5")
#model_n1 = keras.models.load_model('my_GRU_model.h5', custom_objects={'AttentionWithContext':AttentionWithContext()})

#model_n1.compile(Adam(lr=0.01), 'categorical_crossentropy', metrics=['accuracy'])
#model_n1.load_weights("model_doc_tagging_attention_GRU_glove_vec_100d_new.h5")

file = open("tokenizer.pickle",'rb')
tokenizer = pickle.load(file)
#print("len(tokenizer.word_index) + 1:",len(tokenizer.word_index) + 1)
file.close()

# API route
@app.route('/getfile', methods=['GET','POST'])
def getfile():
    if request.method == 'POST':
        labels = ['Bank Bailout', 'Budget', 'Credit Card', 'Family Finance', 
        'Job Benefits', 'Taxes'] 
        
        def cleanSentence(sentence):
             sentence_clean= re.sub("[^a-zA-Z0-9]"," ", str(sentence))
             sentence_clean = sentence_clean.lower()
             tokens = word_tokenize(sentence_clean)
             stop_words = set(stopwords.words("english"))
             sentence_clean_words = [w for w in tokens if not w in stop_words]
             return ' '.join(sentence_clean_words)

        def getTrainSequences(sentence, tokenizer,seq_maxlen):
            sent1 = tokenizer.texts_to_sequences(sentence)
            #sent_maxlen = max([len(s) for s in sent1])
            #seq_maxlen = max([sent_maxlen])
            return np.array(pad_sequences(sent1, maxlen=seq_maxlen))
        #Need pickel files of Tokenizer,Embedding weight and Glove matrix and freezed model

        # for secure filenames. Read the documentation.
        file = request.files['file']
        file.save(os.path.join('C:\\Users\\Atchyuta\\Downloads\\',file.filename))
        with open('C:\\Users\\Atchyuta\\Downloads\\'+ file.filename) as f:
            file_content = f.read().replace('\n', ' ')
        #Lines = file.readlines()
        #file_content= file.read().replace('\n', ' ')
        file_content=re.sub("\d+\.\d+ ", '',file_content)
        #print("in:",file_content)
        #data1=pd.DataFrame()
        #data1['file_content']=file_content
        dict1={"file_content":file_content}
        data1=pd.DataFrame(dict1,index=[0])
        #print(data1['file_content'])
        data1['clean_file_content'] = list(map(cleanSentence, data1['file_content']))
        #print(data1['clean_file_content'])
        corpus_textn = '\n'.join((data1['clean_file_content']))
        sentences1 = corpus_textn.split('\n')
        train_data_need1 = getTrainSequences(sentences1, tokenizer,seq_maxlen=1680)
        #print(train_data_need1.shape)
        #model_n1.load_weights("model_doc_tagging_attention_GRU_glove_vec_100d_new.h5")
        with graph.as_default():
            result=model_n1.predict(train_data_need1)
        #print(result)
        result=result.argmax(axis=-1)
        label= labels[result[0]]
        #print(label)
        

        return flask.jsonify({'predicted_tag': label })  


   
    return "txt file needed"

@app.route("/")
def index():
    return "Index API"


# HTTP Error handlers
@app.errorhandler(404)
def url_error(e):
    return f"""Wrong URL! <pre>{e}</pre>""", 404


@app.errorhandler(500)
def server_error(e):
    return (
        f"""An internal error occured: <pre>{e}</pre>. See logs for full stacktrace""",
        500,
    )


if __name__ == "__main__":
    # This is used when running locally
    app.run(host="0.0.0.0", debug=False)