import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from transformers import RobertaTokenizer, RobertaModel
#loading data
df = load_dataset("pointe77/credit-card-transaction")

#removing irrelevant columns
data=df.remove_columns(['Unnamed: 0',
 'trans_date_trans_time',
 'cc_num',
 'first',
 'last',
 'city_pop',
 'job',
 'dob',
 'gender',
 'unix_time',
 'merch_zipcode',
 'state',
 'zip',
 'lat',
 'long',
 'is_fraud'])

#loading training and testing split
train_data=data['train'].to_pandas()
test_data=data['test'].to_pandas()

#Encoding transaction categories
le = LabelEncoder()
train_data['category'] = le.fit_transform(train_data['category'])

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
m = RobertaModel.from_pretrained('roberta-base')

def generate_embeddings(texts):
    '''
    Generates entity embedding for text in column using roberta model last hidden state
    '''
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = m(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def organizing_embeddings(data, embeddings_dict):
    '''
    Organizing the embeddings
    '''
    embedded_data = []
    for column in cat_columns:
        embedded_column = np.array([embeddings_dict[column][value] for value in data[column]])
        embedded_data.append(embedded_column)
    return np.hstack(embedded_data)


# generating embedding columns
cat_columns=['merchant','city']
train_embeddings_dict=dict()
for column in cat_columns:
    unique_texts = train_data[column].unique()
    embeddings = generate_embeddings(list(unique_texts))
    train_embeddings_dict[column] = dict(zip(unique_texts, embeddings))

train_data_embed = organizing_embeddings(train_data, train_embeddings_dict)
train_data_embed = np.hstack([train_data_embed,train_data['amt'].values.reshape(-1, 1)])
del data

#training
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(train_data['category'].unique()), eval_metric='mlogloss')
model.fit(train_data_embed,train_data['category'])

del train_data_embed

test_embeddings_dict=dict()
for column in cat_columns:
    unique_texts = test_data[column].unique()
    embeddings = generate_embeddings(list(unique_texts))
    test_embeddings_dict[column] = dict(zip(unique_texts, embeddings))

test_data_embed = organizing_embeddings(test_data, test_embeddings_dict)
test_data_embed = np.hstack([test_data_embed,test_data['amt'].values.reshape(-1, 1)])

y_pred = model.predict(test_data_embed)
ct=0
for i,j in zip(test_data['category'],y_pred):
    if i==le.inverse_transform([j]):
        ct+=1
print((ct/len(test_data))*100)
model.save_model('classifier.json')
# import pickle
# pickle.dump(model,open("classifier.pkl","wb"))