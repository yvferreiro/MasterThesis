from gensim.models import fasttext
from gensim.test.utils import datapath

file = "C:/Users/danie/Desktop/crawl-300d-2M-subword.bin"
#file = "C:/Users/danie/Desktop/crawl-300d-2M-subword.bin"

model = fasttext.load_facebook_vectors(datapath(file))
print(model)

word = "cat"
model.most_similar(word)
model.similarity("dog", "cat")
'landlord' in model.key_to_index 

Male_word_list =["boy", "boy’s", "boyhood", "boyish", "boys", "fella", "gent", 
                 "gentleman", "gentleman’s", "gentlemen", "gents", "guy", "guys", 
                 "he", "hes", "him", "himself", "his", "lad", "laddie", "male", 
                 "male’s", "males", "man", "man’s", "manhood", "manly", "masculine", 
                 "masculinity", "men", "mens", "mister", "mr", "schoolboy", "schoolboys", "sir"]

Female_word_list = ["female", "female’s", "females", "feminine", "femininity", 
                    "femme", "gal", "gals", "girl", "girl’s", "girlhood", "girlish", 
                    "girls", "girly", "her", "hers", "herself", "ladies", "lady", 
                    "lady’s", "lass", "lassie", "ma’am", "maam", "madam", "maiden", 
                    "missus", "ms", "schoolgirl", "schoolgirls", "she", "shes", 
                    "woman", "woman’s", "womanhood", "womanly", "women", "womens"]

model.n_similarity(Male_word_list, Female_word_list)
#https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4#15-removed-third-party-wrappers
#link to the gensim model funcitons

from nltk.tokenize import RegexpTokenizer
import pandas as pd
from newsdataapi import NewsDataApiClient
api = NewsDataApiClient(apikey="pub_158807ed74e08f77156e05324333c37f9b917")
response = api.news_api(q= "fashion", language= "en")
results = response['results']
print(results)
article_df = pd.json_normalize(results)
article_df = article_df.assign(Article_Number=range(len(article_df)))
article_df.info()
article_df


article_df['content'] = article_df['content'].astype(str).str.lower()
#print(df[['full_description']].to_string(index=False))

#tokenization
regexp = RegexpTokenizer('\w+')
article_df['text_token']=article_df['content'].apply(regexp.tokenize)
article_df['text_token'][[0]]

model.n_similarity(Male_word_list, article_df['content'].iloc[1])
model.n_similarity(Male_word_list, article_df['text_token'].iloc[1])

model.n_similarity(Male_word_list, article_df['text_token'].iloc[8])
model.n_similarity(Female_word_list, article_df['text_token'].iloc[8])

