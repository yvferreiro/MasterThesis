#packages

import nltk
import re
from genderize import Genderize
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import pickle
import pandas as pd

#Variables(for now):
name_probability_list = {}

filecsv = r"C:\Users\danie\Desktop\bbc_news_01.csv"
article_df_1 = pd.read_csv(filecsv)
article_df = article_df_1.assign(Article_Number=range(len(article_df)))
article_df = article_df.reset_index()
article_df.drop(["publisher", "header_image", "index", "raw_description", "short_description", "uniq_id", "scraped_at"], axis=1)
year = article_df['published_at'].str[:4]
article_df['year']=year

def split_sentences(article, article_id, year):
    pattern = r'(?<=[a-z0-9"]) *[.?!] *(?=[A-Z])'
    article = re.sub(pattern, r'\g<0> ', article)
    sentences = nltk.sent_tokenize(article)
    sentences_with_id = [(sentence, article_id, year) for sentence in sentences]
    return sentences_with_id

sentences_list = []

# add sentences to a new DF along with article ID 
for article, article_id, year in article_df[['description','Article_Number', 'year']].values:
    sentences = split_sentences(str(article), article_id, year)
    sentences_list.extend(sentences)

sentences_df = pd.DataFrame(sentences_list, columns= ['sentences', 'article_id', 'year'])

def PreProcessing (sentence):
    Male_count = 0
    Female_count = 0
    APIcallfail= 0

#regex_cleanup
    sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence)
    sentence = re.sub(r'\<a href', ' ', sentence)
    sentence = re.sub(r'&amp;', '', sentence) 
    sentence = re.sub("\d+", " ", sentence)
    sentence = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sentence)
    sentence = re.sub(r'<br />', ' ', sentence)
    sentence = re.sub(r"\b's\b", '', sentence)

#tokenize
    sentence =  nltk.TweetTokenizer().tokenize(sentence)

#remove small words
    sentence = [ x for x in sentence if len(x) > 2 ]

#tag_and_stem
    tagged_sentence = nltk.tag.pos_tag(sentence)
    lemma = nltk.stem.WordNetLemmatizer()
    pn_tags = {'NNP', 'NNPS'}

    new_words = []
    proper_nouns = []

    for word, tag in tagged_sentence: 
        if tag not in pn_tags: 
            if tag.startswith("V"):
                lemmas = lemma.lemmatize(word, "v")
            else: 
                lemmas = lemma.lemmatize(word)
            new_words.append((lemmas))
        else:
            proper_nouns.append([word, tag])

    sentence = new_words

#name_gender
    #nltk_results = ne_chunk(tagged_sentence)
    nltk_results = ne_chunk(proper_nouns)

    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            if nltk_result.label() == "PERSON":
                name = name.split(' ')[0]
                try: 
                    word_gender = name_probability_list.get(name)
                    if word_gender is None:
                        word_gender = Genderize().get1(name).get('gender')
                        name_probability_list[name] = word_gender
                    if word_gender == 'male':
                        Male_count += 1
                    if word_gender== 'female':
                        Female_count += 1
                except Exception as exception:
                    APIcallfail +=1
            else: 
                sentence.append(name.strip()) #add a tokenize
    
#Lower
    sentence = [x.lower() for x in sentence]

#contractions
    new_text = []
    for word in sentence:
        contraction = contractions.get(word)
        if contraction is None:
            new_text.append(word)
        else:
            for word in contraction.split():
                new_text.append(word)

    sentence = new_text

#gendered_count
    for w in sentence:
        if w in male_list:
            Male_count += 1
        if w in female_list:
            Female_count += 1

#remove_stopwords
    stops = set(stopwords.words("english"))
    sentence = [x for x in sentence if not x in stops]

#remove_leakage
    new_sent = [x for x in sentence if x not in male_list]
    new_sent = [x for x in new_sent if x not in female_list]
    sentence = new_sent

    print(sentence)
    return sentence, Male_count, Female_count, APIcallfail

#if still having memory issues we needd to chunk the dataset 

sentences_df['encoded_sentences'] = sentences_df['sentences'].apply(PreProcessing)

sentences_df_1 = pd.DataFrame(sentences_df["encoded_sentences"].to_list(), columns=['pre_processed_sent','male_count','female_count','apicall_fail'])
result = pd.concat([sentences_df_1, (sentences_df.reset_index(drop=True))], axis=1)

result.drop("encoded_sentences", axis=1)
