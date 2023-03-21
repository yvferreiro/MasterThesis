import pandas as pd
import nltk
nltk.download('punkt')

filename = 'articles.pkl'

article_df = pd.read_pickle(filename)
article_df = article_df.assign(Article_Number=range(len(article_df)))
article_df = article_df.reset_index()
article_df.info()

# tokenize sentences in an article 
def split_sentences(article, article_id):
    sentences = nltk.sent_tokenize(article)
    sentences_with_id = [(sentence, article_id) for sentence in sentences]
    return sentences_with_id

sentences_list = []

# add sentences to a new DF along with article ID 
for article, article_id in article_df[['content','Article_Number']].values:
    sentences = split_sentences(str(article), article_id)
    sentences_list.extend(sentences)

sentences_df = pd.DataFrame(sentences_list, columns= ['sentences', 'article_id'])

sentences_df

# TO-DO: add the data & find the pronouns to determine gender