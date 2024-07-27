import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import RobertaTokenizer, RobertaModel
import os
import pandas as pd
import numpy as np
import joblib
import pickle

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

label_encoder = joblib.load('label_encoder.pkl')
loaded_model = pickle.load(open("xgb.pkl", "rb"))

def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def replace_with_embeddings(data, embeddings_dict):
    embedded_data = []
    for column in cat_columns:
        embedded_column = np.array([embeddings_dict[column][value] for value in data[column]])
        embedded_data.append(embedded_column)
    return np.hstack(embedded_data)

'''
sample data for inference this structure is preferred
sample_data = {
    'merchant': ['fraud_Rippin, Kub and Mann'],
    'city': ['Moravian Falls'],
    'amt': [5]
}
'''
sample_data = {
    'merchant': ['fraud_Rippin, Kub and Mann'],
    'city': ['Moravian Falls'],
    'amt': [5]
}

#PREPROCESSING
cat_columns=['merchant','city']
sample_df = pd.DataFrame(sample_data)
embeddings_dict = dict()

for column in cat_columns:
    unique_texts = sample_df[column].unique()
    embeddings = generate_embeddings(list(unique_texts))
    embeddings_dict[column] = dict(zip(unique_texts, embeddings))

sample_data_embed = replace_with_embeddings(sample_df, embeddings_dict)
sample_data_embed = np.hstack([sample_data_embed,sample_df['amt'].values.reshape(-1, 1)])


#PREDICTING

y_pred = model.predict(xgb.DMatrix(sample_data_embed))

#Final category label list
label=[]
for i in y_pred:
    label.append(label_encoder.inverse_transform([int(i)]))