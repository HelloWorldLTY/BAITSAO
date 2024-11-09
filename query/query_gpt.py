import requests
from bs4 import BeautifulSoup
import html2text
import mygene
import json
import pickle
mg = mygene.MyGeneInfo()

import pandas as pd
import scanpy as sc

df = pd.read_csv("allgene_name_ensemble.csv", index_col = 0)

gene_all = df['allgene_human'].values

import openai
import time
delay_sec = 5
# remember to set your open AI API key!
openai.api_key = ''

import numpy as np

EMBED_DIM = 1536 # embedding dim from GPT-3.5
lookup_embed = np.zeros(shape=(len(gene_all),EMBED_DIM))

def get_gpt_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return np.array(openai.Embedding.create(input = [text], model=model)['data'][0]['embedding'])

# df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/MARSY/data/split_data.csv", index_col=0)
df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/synergy_prediction/labels_synergy_value.csv")

df_grountruth_score.head()

import pickle
with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_deepsynergycellline.pickle", 'rb') as f:
    cellline_name_getembedding = pickle.load(f)
with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_deepsynergydrugcombineline.pickle", 'rb') as f:
    drug_name_getembedding = pickle.load(f)

all_item_list = []

for item in df_grountruth_score.index:
    d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
    wait_list = [d1, d2, cl]
    all_item_list.append(wait_list)

len(all_item_list)

all_item_list[0]

prompt = 'Please compute the drug synergetic score of the drug combinations 5-FU and ABT-888 based on cell line A2058 using the Loewe rule. The score is: '

drug_name_to_GPT_response = []

gene_completion_test = all_item_list
for idx, gene in enumerate(gene_completion_test):
    print(gene)
    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", 
                    messages=[{"role": "user", 
                               "content": f"Please compute the drug synergetic score of the drug combinations {gene[0]} and {gene[1]} based on cell line {gene[2]} using the Loewe rule. If you do not know then output NA. The score is: "}])
        drug_name_to_GPT_response.append(completion.choices[0].message.content)
        time.sleep(1)
    except (openai.APIError, 
                    openai.error.APIError, 
                    openai.error.APIConnectionError, 
                    openai.error.RateLimitError, 
                    openai.error.ServiceUnavailableError, 
                    openai.error.Timeout) as e:
        #Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    if idx == 100:
        break

len(drug_name_to_GPT_response)

drug_name_to_GPT_response

with open('ensem_desc_baitsao3.5output.pickle', 'wb') as handle:
    pickle.dump(drug_name_to_GPT_response, handle, protocol=pickle.HIGHEST_PROTOCOL)
