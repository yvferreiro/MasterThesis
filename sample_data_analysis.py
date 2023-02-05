#import newsdataapi
from newsdataapi import NewsDataApiClient
import pandas as pd
#import nltk as nltk
from nltk.tokenize import RegexpTokenizer
#import json
#import ast
import re

api = NewsDataApiClient(apikey="pub_158807ed74e08f77156e05324333c37f9b917")

#this page method is what current documentation asks for in the repo but something is wrong 
#they have a bug that makes it want a string and an int at the same time. 

#emailed Naveen and he's on it!
response = api.news_api(q= "covid", language= "en")

#So I'm thinking that we make a few funtions
    #function to call the API (maybe in it's own folder on git)
    #function to take the new things added to the list of articles to the article df
    #function to take all new articles and generate the sentences and words. 
    
    #then we need a whole new file in a folder for each function so we can run them seperatly

#response = api.news_api( q= "fish" , country = "us",page=2)
#Generates the article database
results = response['results']
article_df = pd.json_normalize(results)
article_df = article_df.assign(Article_Number=range(len(article_df)))

#creates a copy of the articles and seperates by sentence 
df_copy = article_df
sentence_df = df_copy.drop(['title', 'link','keywords', 'creator','video_url', 
                            'description', 'pubDate', 'image_url', 'source_id', 
                            'category', 'country', 'language'], axis=1) 

regex_sentence_splitter = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
sentence_df['content'] = article_df['content'].apply(lambda x: re.split(regex_sentence_splitter, x))
sentence_df = sentence_df.explode('content', ignore_index=True)
sentence_df.rename(columns={"Unnamed: 0": "Article_Number"}, inplace=True)
sentence_df.index.name = "Sentence ID"
print(sentence_df)

#takes the sentence df and makes a word df out of it
df_copy2 = sentence_df
word_df = df_copy2.drop(['Article_Number'], axis=1) 

regex_word_splitter = r"\s+"
word_df['content'] = sentence_df['content'].apply(lambda x: re.split(regex_word_splitter, x))
#Would it be smart to add a counter to the articel database here?
word_df = word_df.explode('content', ignore_index=True)
word_df.rename(columns={"Unnamed: 0": "Sentence_Number"}, inplace=True)
word_df.index.name = "Word ID"
#missing the sentence ID. I think I messed something up but i"m going to charge ahead


#This should all be a function. I will change it ASAP

#I used this website to help with this section https://www.kirenz.com/post/2021-12-11-text-mining-and-sentiment-analysis-with-nltk-and-pandas-in-python/text-mining-and-sentiment-analysis-with-nltk-and-pandas-in-python/
#this makes the article text all lower case
df['full_description'] = df['full_description'].astype(str).str.lower()
#print(df[['full_description']].to_string(index=False))

#tokenization
regexp = RegexpTokenizer('\w+')
df['text_token']=df['full_description'].apply(regexp.tokenize)

#stopwords but I'm not currently using this. I'm going to use words that might be on this list.
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#stopwords = nltk.corpus.stopwords.words("english")
#df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords])

df["Word Count"] = df["text_token"].str.len()

his_w = ["his", "he", "man", "uncle", "dad", "father", "boy", "husband"]

from collections import Counter

df["His Count"] = (
    df['full_description'].str.split()
    .apply(Counter)
    .apply(lambda counts: sum(word in counts for word in his_w))
)

her_w = ["her", "she", "woman", "aunt", "mother","mom", "girl", "wife"]


from collections import Counter

df["Her Count"] = (
    df['full_description'].str.split()
    .apply(Counter)
    .apply(lambda counts: sum(word in counts for word in her_w))
)


Summarydf = df[['source_id','Her Count','His Count', 'Word Count']]
Summarydf = Summarydf.groupby(['source_id']).sum()

Summarydf = Summarydf[~(Summarydf['Word Count'] <= 10)]  

Summarydf['Word Difference'] = Summarydf['Her Count']- Summarydf['His Count']
Summarydf['Gendered Proportion'] = (Summarydf['Word Difference']/ Summarydf['Word Count'])*100

Summary = Summarydf.drop(columns = ['His Count', 'Her Count', 'Word Difference','Word Count'])
Summary = Summary.sort_values(by=['Gendered Proportion'])
print(Summary)

