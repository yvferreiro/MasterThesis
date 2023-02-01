import newsdataapi
from newsdataapi import NewsDataApiClient
import pandas as pd
import nltk as nltk
from nltk.tokenize import RegexpTokenizer


api = NewsDataApiClient(apikey="pub_158807ed74e08f77156e05324333c37f9b917")
# FML IT COMES OUT WITH ' not " FUCK
response = api.news_api(country = "us")
print(response)
response2 = str(response)



del response['status']
del response['totalResults']
del response['nextPage']

print(response)

# when downloading a json file from newsio they include a summary of the query. So
#it will say how many results etc. This needs to be replaced with just "data:"
# to allow the df to parse properly. There is even a part that must be removed at the end

# this counts how many news sources are present in the file. 
import ast
parsed_json = ast.literal_eval(response2)
df = pd.json_normalize(parsed_json, record_path=[list(parsed_json)[0], 1])

df = pd.read_json (response2)
pd.json_normalize(response2)
print(response2)
n = len(pd.unique(df['source_id']))

#I cannot get this api key bullshit to actually work. Maybe the bit lab is the answer?





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

