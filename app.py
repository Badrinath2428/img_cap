from flask import Flask, render_template, request
import string

 

from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3


import tensorflow.keras.preprocessing.image
import pickle
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Embedding, 
    TimeDistributed, Dense, RepeatVector, 
    Activation, Flatten, Reshape, concatenate,  
    Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

START = "startseq"
STOP = "endseq"
EPOCHS = 10
USE_INCEPTION = True
max_length=34
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
embedding_dim = 200
vocab_size=1652


with open('wordtoidx.pkl', 'rb') as f:
    wordtoidx = pickle.load(f)

with open('idxtoword.pkl', 'rb') as f1:
    idxtoword = pickle.load(f1)

from tensorflow.keras import models
caption_model=models.load_model('model.h5')

with open('test2048.pkl', "rb") as fp2:
    encoding_test = pickle.load(fp2)

preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input
encode_model=models.load_model('encode_model.h5',compile=False)

def encodeImage(img):
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    x = tensorflow.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = encode_model.predict(x)
    x = np.reshape(x, OUTPUT_DIM )
    return x

def generateCaption(photo):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final



app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    img = request.files['file1']
    img.save('static/file.jpg')
    print("="*50)
    print("IMAGE SAVED")

    img = tensorflow.keras.preprocessing.image.load_img('static/file.jpg',target_size=(HEIGHT, WIDTH))
    test= encodeImage(img).reshape((1,OUTPUT_DIM))
    print(test.shape)
    print("Caption:",generateCaption(test))
    print("_____________________________________")
    return render_template('predict.html', data=generateCaption(test))

if __name__ == "__main__":
    app.run(debug=True)
