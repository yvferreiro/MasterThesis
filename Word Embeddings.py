from gensim.models import fasttext
from gensim.test.utils import datapath
import nltk

file = "C:/Users/danie/Desktop/crawl-300d-2M-subword.bin"
#file = "C:/Users/danie/Desktop/crawl-300d-2M-subword.bin"

from newsdataapi import NewsDataApiClient
import pandas as pd
api = NewsDataApiClient(apikey="pub_158807ed74e08f77156e05324333c37f9b917")
response = api.news_api(q= "fashion week", language= "en", country = "us")
results = response['results']
article_df = pd.json_normalize(results)

#function that will the articles we pull using the API 
def save_article_df(article_df, filename):
    try:
        saved_df = pd.read_pickle(filename)
    except FileNotFoundError:
        saved_df = pd.DataFrame()
    saved_df = saved_df.append(article_df)
    saved_df.to_pickle(filename)

#save article in a pickle file 
filename = "articles.pkl"
save_article_df(article_df, filename)
article_df = pd.read_pickle(filename)
article_df = article_df.dropna(subset=['content'])
article_df['content']

#regexp = RegexpTokenizer('\w+')
article_df['content'] = article_df['content'].astype('string')
#article_df['text_token']= article_df['content'].apply(regexp.tokenize)

article_df['text_token'] = article_df.apply(lambda row: nltk.word_tokenize(row['content']), axis=1)

##IDENTIFY ADJECTIVES 
# Define a function to extract the adjectives from a text string
def get_adjectives(text_input):
    tokens = nltk.word_tokenize(text_input)
    pos = nltk.pos_tag(tokens)
    adjectives = [word for word, tag in pos if tag.startswith('JJ')]
    return str(adjectives)

# Apply the function to each text string in the series
adjectives_series = article_df['content'].apply(get_adjectives)
print(str(adjectives_series)) #transform into string for future operations 

#function clean up resulting adjectives 
def adj_complete_series(adjective_serie):
    adj_clean = adjective_serie.str.replace(r'\b\w\b|[^\w\s]', '', regex=True) #replace single characters and whitespaces with empty string
    adj_clean = adj_clean.str.strip().str.split('\s+') #remove whitespace
    adj_clean = adj_clean.apply(lambda x: [word for word in x if len(word) > 1]) #remove single letter word
    return adj_clean

adjective_complete = adj_complete_series(adjectives_series)
adjective_complete #list of adjectives identified in each article is ready 


##METHOD 1: PRE-TRAINED FASTTEXT (REALLY BASIC) ------------------------------------------------------------------------
model = fasttext.load_facebook_vectors(datapath(file))

# The list of male and female words from the People = Men paper
Male_word_list =["boy", "boy’s", "boyhood", "boyish", "boys", "fella", "gent", "gentleman", "gentleman’s", "gentlemen", "gents", "guy", "guys", "he", "hes", "him", "himself", "his", "lad", "laddie", "male", "male’s", "males", "man", "man’s", "manhood", "manly", "masculine", "masculinity", "men", "mens", "mister", "mr", "schoolboy", "schoolboys", "sir"]
Female_word_list = ["female", "female’s", "females", "feminine", "femininity", "femme", "gal", "gals", "girl", "girl’s", "girlhood", "girlish", "girls", "girly", "her", "hers", "herself", "ladies", "lady", "lady’s", "lass", "lassie", "ma’am", "maam", "madam", "maiden", "missus", "ms", "schoolgirl", "schoolgirls", "she", "shes", "woman", "woman’s", "womanhood", "womanly", "women", "womens"]

model.n_similarity(Male_word_list, Female_word_list)

#function to calculate MALE similarity of the content of any article 
def calculate_male_similarity(row):
    article_words = row['text_token']
    return model.n_similarity(article_words, Male_word_list)

#function to calculate FEMALE similarity of the content of any article 
def calculate_female_similarity(row):
    article_words = row['text_token']
    return model.n_similarity(article_words, Female_word_list)

#calculate the distance between male and female similarity 
article_df['female_similarity'] = article_df.apply(calculate_female_similarity, axis=1)
article_df['male_similarity'] = article_df.apply(calculate_male_similarity, axis=1)
article_df['malefemale_diff'] = article_df['male_similarity']-article_df['female_similarity'] 
pd.set_option('display.max_rows', None)




## METHOD 2: PRE-TRAINED GloVe (Global Vectors) ------------------------------------------------------------------------
import gensim.downloader as api

# Load the pre-trained GloVe vectors with 100 dimensions
glove_vectors = api.load("glove-wiki-gigaword-100")
# Find the most similar words to "sexy"
similar_words = glove_vectors.most_similar("sexy")
# Print the results
for word, similarity in similar_words:
    print(f"{word}: {similarity}")

# important: the model below is too big to train, but the code works when tested on one word 
# TO-DO - I think we should write a function identifying "gendered" words in the text and then finding the similarity to those
# Load the pre-trained GloVe vectors with 100 dimensions
glove_vectors = api.load("glove-wiki-gigaword-100")
# Define a list of words for male & female 
Male_word_list #from people = men 
Female_word_list #from people = men 
# Find the most similar words to each word in the list
similar_words_male = {}
for word in Male_word_list:
    similar_words_male[word] = glove_vectors.most_similar(word)

similar_words_female = {}
for word in Female_word_list:
    similar_words_female[word] = glove_vectors.most_similar(word)

# Print the results
for word, similar_word_list_female in similar_words_female.items():
    print(f"Similar words to {word}:")
    for similar_word, similarity in similar_word_list_female:
        print(f"- {similar_word}: {similarity}")