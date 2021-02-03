import glob
import os
import pickle
import random
import numpy as np
from keras import Input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.layers.merge import add
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array

# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ------------------ Functions ------------------

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# --------------------------- Data processing --------------------------

# %matplotlib inline

description_path = "data/titles.txt"
filenames_path = "data/filenames.txt"
images_path = 'data/Images/'
glove_path = 'data/glove/'

word_count_threshold = 2


new_descriptions = load_doc(description_path)

doc = open(filenames_path,'r').read()
dataset = list()
for line in doc.split('\n'):
    if len(line) > 1:
      identifier = line.split('.')[0]
      dataset.append(identifier)

train = set(dataset)

img = glob.glob(images_path + '*.jpg')
train_images = set(open(filenames_path, 'r').read().strip().split('\n'))
train_img = []

print(train_images)

for i in img:
    if i[len(images_path):] in train_images:
        train_img.append(i)
print(train_img)


train_descriptions = dict()
for line in new_descriptions.split('\n'):
    tokens = line.split()
    image_id, image_desc = tokens[0], tokens[1:]
    if image_id in train:
        if image_id not in train_descriptions:
            train_descriptions[image_id] = list()
        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        train_descriptions[image_id].append(desc)

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

vocab.remove('vanitas')
vocab.remove('stilllife')


print('Vocabulary = %d' % (len(vocab)))

ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1

all_desc = list()
for key in train_descriptions.keys():
    [all_desc.append(d) for d in train_descriptions[key]]
lines = all_desc
max_length = max(len(d.split()) for d in lines)

print('Description Length: %d' % max_length)

# ------------------------------------- Create embeddings ---------------------

embeddings_index = {}
f = open(os.path.join(glove_path, 'glove.6B.200d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

pickle.dump(ixtoword, open('data/ixtoword.p', 'wb'))
pickle.dump(wordtoix, open('data/wordtoix.p', 'wb'))
np.save('data/embedding_matrix.npy', embedding_matrix)
