from gensim.models import fasttext
from gensim.test.utils import datapath
import nltk

file = "C:/Users/danie/Desktop/crawl-300d-2M-subword.bin"
#file = "C:/Users/danie/Desktop/crawl-300d-2M-subword.bin"

model = fasttext.load_facebook_vectors(datapath(file))

# The list of male and female words from the People = Men paper
Male_word_list =["boy", "boy’s", "boyhood", "boyish", "boys", "fella", "gent", "gentleman", "gentleman’s", "gentlemen", "gents", "guy", "guys", "he", "hes", "him", "himself", "his", "lad", "laddie", "male", "male’s", "males", "man", "man’s", "manhood", "manly", "masculine", "masculinity", "men", "mens", "mister", "mr", "schoolboy", "schoolboys", "sir"]
Female_word_list = ["female", "female’s", "females", "feminine", "femininity", "femme", "gal", "gals", "girl", "girl’s", "girlhood", "girlish", "girls", "girly", "her", "hers", "herself", "ladies", "lady", "lady’s", "lass", "lassie", "ma’am", "maam", "madam", "maiden", "missus", "ms", "schoolgirl", "schoolgirls", "she", "shes", "woman", "woman’s", "womanhood", "womanly", "women", "womens"]

model.n_similarity(Male_word_list, Female_word_list)

from newsdataapi import NewsDataApiClient
import pandas as pd
api = NewsDataApiClient(apikey="pub_158807ed74e08f77156e05324333c37f9b917")
response = api.news_api(q= "fashion week", language= "en", country = "us")
results = response['results']
article_df = pd.json_normalize(results)

def save_article_df(article_df, filename):
    try:
        saved_df = pd.read_pickle(filename)
    except FileNotFoundError:
        saved_df = pd.DataFrame()
    saved_df = saved_df.append(article_df)
    saved_df.to_pickle(filename)

filename = "articles.pkl"
save_article_df(article_df, filename)
article_df = pd.read_pickle(filename)
article_df = article_df.dropna(subset=['content'])
article_df['content']

#regexp = RegexpTokenizer('\w+')
article_df['content'] = article_df['content'].astype('string')
#article_df['text_token']= article_df['content'].apply(regexp.tokenize)

article_df['text_token'] = article_df.apply(lambda row: nltk.word_tokenize(row['content']), axis=1)

def calculate_male_similarity(row):
    article_words = row['text_token']
    return model.n_similarity(article_words, Male_word_list)

def calculate_female_similarity(row):
    article_words = row['text_token']
    return model.n_similarity(article_words, Female_word_list)

article_df['female_similarity'] = article_df.apply(calculate_female_similarity, axis=1)
article_df['male_similarity'] = article_df.apply(calculate_male_similarity, axis=1)
article_df['malefemale_diff'] = article_df['male_similarity']-article_df['female_similarity'] 
pd.set_option('display.max_rows', None)

