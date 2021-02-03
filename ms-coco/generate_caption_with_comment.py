import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import string, os, glob, pickle
from PIL import Image
from time import time
from keras import Input, layers
from keras import optimizers
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
import random
from keras.utils import to_categorical

# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# pic_to_predict = 'imgs/example_1.jpg'
model_weights = 'weights/model_coco_30.hdf5'


# Loading prepared files and set hyperparams
ixtoword = pickle.load(open('data/ixtoword.p', 'rb'))
wordtoix = pickle.load(open('data/wordtoix.p', 'rb'))
embedding_matrix = np.load('data/embedding_matrix.npy')
vocab_size = len(ixtoword) + 1
max_length = 22
embedding_dim = 200

# Define Inception model
model_inception_complete = InceptionV3(weights='imagenet')
model_inception_notop = Model(model_inception_complete.input, model_inception_complete.layers[-2].output)

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image)
    # fea_vec = model_inception_notop.predict(image)
    # fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return image

# Define image caption model
# inputs1 = Input(shape=(2048,))
inputs1 = model_inception_complete.input

# fe1 = Dropout(0.5)(inputs1)
fe1 = Dropout(0.5)(model_inception_complete.layers[-2].output)

fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.load_weights(model_weights, by_name=True)

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def beam_search_predictions(image, beam_index=3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model.predict([image, par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

linking_sentences_1 = ["The galleries seem to be full of ... ",
                       "This looks like ehm ...",
                       "In Italy people call this ... ",
                       "OMG, this is like ... ",
                       "Should we say this looks like ... ",
                       "People make strange objects like this ... ",
                       "Is this love or is this a ... ",
                       "Some curators would say that this is ... ",
                       "In this world this could be ...",
                       "Baby ... is this ... ",
                       "Object oriented humans think this is ... "]

linking_sentences_2 = [" ... but my knowledge tells me this could be ... ",
                       " ... but my unhumble experience would say ... ",
                       " ... but I would say this is hmmm ... ",
                       " ... but from the context I understand this is ... ",
                       " ... but my refined taste tells me that this is ... ",
                       " ... but no things can be concluded by refined taste, thats why I think this is ... ",
                       " ... but some curators would say this is ... ",
                       " ... but in her world this is ... ",
                       " ... or is it more like ehhh ... ",
                       " ... but this object looks also like ... ",
                       " ... but La Contextressa thinks this is ... "]

# image = encode(pic_to_predict).reshape((1,2048))
for image_to_predict in sorted(os.listdir('./imgs')):
    loaded_image = encode('./imgs/'+image_to_predict)

    x=plt.imread('./imgs/'+image_to_predict)
    plt.imshow(x)
    plt.show()

    # print("Greedy Search start")
    # print("Greedy Search:",greedySearch(loaded_image))
    # print("Beam Search, K = 3:",beam_search_predictions(loaded_image, beam_index = 3))
    # print("Beam Search, K = 5:",beam_search_predictions(loaded_image, beam_index = 5))
    # print("Beam Search, K = 7:",beam_search_predictions(loaded_image, beam_index = 7))
    # print("Beam Search, K = 10:",beam_search_predictions(loaded_image, beam_index = 10))
    print(f'{image_to_predict}: {random.choice(linking_sentences_1)} '
          f'{beam_search_predictions(loaded_image, beam_index = 6)} {random.choice(linking_sentences_2)}')
    # print(f'{image_to_predict}: {random.choice(linking_sentences_1)} '
    #       f'{greedySearch(loaded_image)} {random.choice(linking_sentences_2)}')


