import json, random, pprint, string
import pandas as pd
import numpy as np
import time

with open('captions_train2014.json') as json_file:
    dict = json.load(json_file)
    json_file.close()

# with open('captions_val2014.json') as json_file:
#     dict_val = json.load(json_file)
#     json_file.close()


def clean_descriptions(desc):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    # for key, desc_list in descriptions.items():
    #     for i in range(len(desc_list)):
    # desc = desc_list[i]
    # tokenize
    desc = desc.split()
    # convert to lower case
    desc = [word.lower() for word in desc]
    # remove punctuation from each token
    desc = [w.translate(table) for w in desc]
    # remove hanging 's' and 'a'
    desc = [word for word in desc if len(word) > 1]
    # remove tokens with numbers in them
    desc = [word for word in desc if word.isalpha()]
    # store as string
    return ' '.join(desc[:min(len(desc),20)])

data = pd.DataFrame(dict.get('annotations'))

data = data[['image_id', 'caption']]

# data = data.head(1000)

filenames = data['image_id'].copy()

start = time.time()

for line, row in data.iterrows():
    data.loc[line, 'image_id'] = 'COCO_train2014_000000'+str(data.loc[line, 'image_id']).zfill(6)
    filenames.loc[line] = data.loc[line, 'image_id']+'.jpg'
    if isinstance(data.loc[line, 'caption'], pd.Series):
        data.loc[line, 'caption'] = clean_descriptions(data.loc[line, 'caption'].to_string())
    else:
        data.loc[line, 'caption'] = clean_descriptions(data.loc[line, 'caption'])
    if line % 5000 == 0:
        end = time.time()
        print(f'Finished {line} rows out of {len(data.index)} in time {end - start}')

data = data.sort_values('image_id')
filenames.sort_values(inplace=True)
filenames = filenames.drop_duplicates()

np.savetxt('./data/descriptions.txt', data.values, fmt='%s')
np.savetxt('./data/filenames.txt', filenames.values, fmt='%s')


