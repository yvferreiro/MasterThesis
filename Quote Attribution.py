from newsdataapi import NewsDataApiClient
import pandas as pd
import pickle 
import re
from nltk import tokenize

verbs= ["acknowledge", " add", " address", " admit", " announce", " argue", " believe", " claim", " conclude", " confirm", " continue", " declare", " describe", " ensure", " estimate", " explain", " find", " indicate", " inform", " insist", " note", " point", " predict", " provide", " release", " reply", " report", " respond", " say", " state", " suggest", " tell", " testify", " think", " tweet", " warn", " write"]


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

article_df = article_df.assign(Article_Number=range(len(article_df)))
article_df = article_df.reset_index()
sample_article = article_df["content"].loc[42]
sample_article
quote_db= pd.DataFrame(columns=["article_number","quote", "quote_loc"])

def find_quotes(df):
    df = df.replace('“','"').replace('”','"')
    quotes = re.findall(r'["\']([^"\']*)["\']', df)
    indirect_quotes = ([x for x in tokenize.sent_tokenize(df) if any(y in x for y in verbs)])
    quotes= quotes + indirect_quotes
    quote_location = []
    for word in quotes:
        word_loc = df.find(word)
        word_len = len(word)
        word_loc2 = word_loc + word_len
        quote_location.append((word_loc, word_loc2))
    return quotes, quote_location

def attribute_quotes(quotes, df):
    

find_quotes(sample_article)
quote_db
