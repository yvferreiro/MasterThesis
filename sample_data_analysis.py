import pandas as pd
import nltk as nltk
from nltk.tokenize import RegexpTokenizer

# when downloading a json file from newsio they include a summary of the query. So
#it will say how many results etc. This needs to be replaced with just "data:"
# to allow the df to parse properly. There is even a part that must be removed at the end

# this counts how many news sources are present in the file. 

df = pd.read_json ('sample_dataset.json', orient='split')
n = len(pd.unique(df['source_id']))
print(n)

#I used this website to help with this section https://www.kirenz.com/post/2021-12-11-text-mining-and-sentiment-analysis-with-nltk-and-pandas-in-python/text-mining-and-sentiment-analysis-with-nltk-and-pandas-in-python/

#this makes the article text all lower case
df['full_description'] = df['full_description'].astype(str).str.lower()
print(df[['full_description']].to_string(index=False))

#tokenization
regexp = RegexpTokenizer('\w+')
df['text_token']=df['full_description'].apply(regexp.tokenize)
print(df[['text_token']].to_string(index=False))

#stopwords but I'm not currently using this. I'm going to use words that might be on this list.
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#stopwords = nltk.corpus.stopwords.words("english")
#df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords])

