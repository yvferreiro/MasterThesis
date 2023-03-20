from newsdataapi import NewsDataApiClient
import pandas as pd
import pickle 
import re
from nltk import tokenize

reporting_words = ["acknowledge", " add", " address", " admit", " announce", " argue", " believe", " claim", " conclude", " confirm", " continue", " declare", " describe", " ensure", " estimate", " explain", " find", " indicate", " inform", " insist", " note", " point", " predict", " provide", " release", " reply", " report", " respond", " say", " state", " suggest", " tell", " testify", " think", " tweet", " warn", " write"]
 If sent contains a reporting word then
    look to the word before for a name or a pronoun
        If found and a name find gender 
            once gender is found clasify the sentence
        if found a pronoun then classify sentence
        else   
            look after the word for the same and follow the same pattern
                else follow the rest of the pattern
Count the number of male, female, and neutral pronouns

*****Look to see if the sentence contains a proper noun and add the gender to the ocunt
    I think this should be added to the preprocessing because we are already looking up the POS tags so it will use less bullshit


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
sample_article = article_df["content"].loc[57]
sample_article
article_df.drop("index",axis=1)
quote_df= pd.DataFrame(columns=["quote", "quote_loc", "previous_sent", "next_sent"])

def find_quotes(df):
    df = df.replace('“','"').replace('”','"')
    quotes = re.findall(r'((?:[^.]*?\.){0,2})\s*([^"]*"[^"]*")\s*((?:\.[^.]*?){0,2})', df)
    for quote in quotes:
        quote_df = quote_df.append({"previous_sent": quote[0], "quote": quote[1], "next_sent": quote[2]}, ignore_index=True)
    #quotes = re.findall(r'["\']([^"\']*)["\']', df)
    quotes= [x for x in quotes if x.count(" ") > 3]
    quote_location = []
    for word in quotes:
        word_loc = df.find(word)
        word_len = len(word)
        word_loc2 = word_loc + word_len
        quote_location.append((word_loc, word_loc2))
    return quotes, quote_location

def find_indirect_quotes(df):
    indirect_quotes = ([x for x in tokenize.sent_tokenize(df) if any(y in x for y in verbs)])
    indirect_quotes = [ x for x in indirect_quotes if '"' not in x ]
    quote_location = []
    for word in indirect_quotes:
        word_loc = df.find(word)
        word_len = len(word)
        word_loc2 = word_loc + word_len
        quote_location.append((word_loc, word_loc2))
    return indirect_quotes, quote_location

    #quotes= quotes + indirect_quotes

#def attribute_quotes(quotes, df):
    

find_quotes(sample_article)

quote_df= pd.DataFrame(columns=["prev_sentence", "quote_sentence", "next_sentence"])

def find_quotes_2(news_article):
    news_article = news_article.replace('“','"').replace('”','"')
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    sentences = re.split(sentence_pattern, news_article)
    print(sentences)
    quote_pattern = r'"(.+?)"'

    matched_sentences = []

    for i in range(len(sentences)):
        if i == 0:
            prev_sentence = None
        else:
            prev_sentence = sentences[i-1]  
        curr_sentence = sentences[i]
        if i == len(sentences)-1:
            next_sentence = None
        else:
            next_sentence = sentences[i+1]
        
        # Check if current sentence contains a quote
        if re.search(quote_pattern, curr_sentence):
            quote = re.search(quote_pattern, curr_sentence).group()
            if len(re.findall(r'\w+', quote)) >= 4:
                matched_sentences.append({
                    'prev_sentence': prev_sentence,
                    'quote_sentence': curr_sentence,
                    'next_sentence': next_sentence
                })

    df = pd.DataFrame(matched_sentences)
    return df
        
find_quotes_2(sample_article)

