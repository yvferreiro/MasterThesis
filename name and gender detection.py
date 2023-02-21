from newsdataapi import NewsDataApiClient
import pandas as pd
import seaborn as sns
import numpy as np
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from genderize import Genderize
#import re

api = NewsDataApiClient(apikey="pub_158807ed74e08f77156e05324333c37f9b917")
response = api.news_api(q= "election", language= "en", country = "us")
results = response['results']
article_df = pd.json_normalize(results)
article_df = article_df.assign(Article_Number=range(len(article_df)))

#made some tweaks here for smoother working 
article_df["creator"] = article_df["creator"].astype('string').replace("'", '', regex=True) #remove speech marks
article_df["creator"] = article_df["creator"].str.strip('[]') #remove square brackets
article_df["creator"] = article_df["creator"].fillna("None ") #added this so that NaNs won't be an issue in future ops
article_df['pos'] = article_df['creator'].str.find(' ')
article_df['author_first_name'] = article_df.apply(lambda x: x['creator'][0:x['pos']], axis=1)

# functiont save articles to a file 
import pickle 
def save_article_df(article_df, filename):
    try:
        # Load existing dataframe if file exists
        saved_df = pd.read_pickle(filename)
    except FileNotFoundError:
        # Create a new empty dataframe if file doesn't exist
        saved_df = pd.DataFrame()

    # Append the new article dataframe to the saved dataframe
    saved_df = saved_df.append(article_df)

    # Save the updated dataframe to the file
    saved_df.to_pickle(filename)


#save the article 
#article_df = pd.DataFrame({'title': ['Article 1'], 'content': ['This is a sample article.']})
filename = "articles.pkl"
save_article_df(article_df, filename)


#this is how we had to write gender detection
#article_df["creator"] = article_df["creator"].astype('string')
#article_df['creator'] = article_df['creator'].apply(lambda x: x.replace('[','').replace(']','').replace("'",'')) 
#article_df['pos'] = article_df['creator'].str.find(' ')
#article_df['author_first_name'] = article_df.apply(lambda x: x['creator'][0:x['pos']],axis=1)
#article_df['author_first_name']

#maybe for this function we need to do something to make the naming unique 
def name_function(df_name, names):
    list_a = []
    for x in names:
        if x == "None":
            list_a.append(0)
        else: 
            nltk_results = ne_chunk(pos_tag(word_tokenize(x)))
            list_b = []
            for nltk_result in nltk_results:
                if type(nltk_result) == Tree:
                    name = ''
                    for nltk_result_leaf in nltk_result.leaves(): 
                        name += nltk_result_leaf[0]
                        list_b.append([nltk_result.label(), name])
            list_a.append(list_b)
    return list_a

named_individuals = name_function(article_df, article_df['creator'])

#This function prints the categories of name of person (based on 1st name), 1st name of person identified 
#for the outputs of "name_individuals", a variable created to capture the output of function "named_function". 
#Important note: I plug in first and last names because if we only plug in first names, it identifies people as GPE when it is not the case. 

names3_cat = []
names3_name = []

#This function prints the categories of name of person (based on 1st name), 1st name if person identified, and then creates a list with 1-hot encoding 
#for the outputs of "name_individuals", a variable created to capture the output of function "named_function". 
#Important note: I plug in first and last names because if we only plug in first names, it identifies people as GPE when it is not the case. 

def extract_cat(lst):
    for j in range(len(lst)): #list of list (list of first name + labels and last name + labels)
        if len(lst[j]) == 0: 
            names3_cat.append("cannot categorize person or organization")
            names3_name.append("None ")
        elif len(lst[j]) == 2: #if len > 1 then there is a label and a name
            names3_cat.append([item[0] for item in lst[j]])
            names3_name.append([item[1] for item in lst[j]])
        else: 
            names3_cat.append("no label identified")
            names3_name.append("no person identified")

    return names3_cat, names3_name

#confirms almost all individuals are picked up as person
names2 = extract_cat(named_individuals) 

#Create a new DF storing 1st names and gender (independent)
names2_list_cat = list((names2)[0])
names2_list_name = list((names2)[1])

#function to grab category of 1st name 
first_cat = []
def cats1(categories): 
    for i in range(len(categories)):
        if len(categories[i]) == 2: 
            first_cat.append(categories[i][0])
        else: 
            first_cat.append("not a person")
    return first_cat #cat is a list

cat1 = cats1(names2_list_cat) #name category variable "cat1"

#function to only grab first from a list with first and last name 
def extracto_first_name(column):
    """
    This function takes in a pandas column and outputs the first value of a list
    if the item passed for a row is a list and output the value of the item if it is not.
    """
    return column.apply(lambda x: x[0] if isinstance(x, list) else x)



#create DF with labels, names, and categories
df_1 = pd.DataFrame(np.column_stack([cat1, names2_list_name]), columns = ["Label", "Names"], dtype = object)
names2_list_name = extracto_first_name(df_1["Names"])

#apply genderize function to the author first name column in article_df
def genderize(name): 
    if name == "None": #None means there is no name available 
        return "not identifiable"
    elif name == "CBS" or name == "ABC": #manually weed out news publications 
        return "not identifiable"
    else: 
        return Genderize().get([name])[0]["gender"]
article_df['creator gender']= article_df.apply(lambda row : genderize(row["author_first_name"]),axis=1)
article_df[['creator', "author_first_name", 'creator gender']]

## Note: because it picks out media publications as "males", I added some lines of code to manually weed out 
## publication names that have been occurring. 

#function to get the gender of the first name 
#def genderize(name): 
#    return Genderize().get([name])[0]["gender"]
article_df['creator gender']= article_df.apply(lambda row : genderize(row["author_first_name"]),axis=1)
print(article_df['creator gender'])
print(article_df['creator'])

## COMMENTS ON GENDERIZE: 
# (a) I read the documentation and it seems that Genderize is fully capable of identifying non-Western names. 
# As part of its parameters, you can input the ISO country or language code to tailor the search of names in that place
# or in that language. It is cool, but I am not sure if we want to specify that given how international the range of folks reporting
# in English is. 
# (b)Genderize does not have the ability to check for the gender of a combine name. It scans through each part of the item and attributes gender. 
#(c) The limit of names per API request is 10, so that is in line with the news data.  


#EDA get count of male v. female creators 
sns.set_theme(style="whitegrid")
article_df['creator gender'].value_counts().plot(kind='bar', color = ["tomato", "skyblue", "grey"])

#Notes 
# The functions get confused when there are multiple authors, we need to figure out how to deal with this 
# New DF with name and associated gender has been created --> "df_1"
# New function was added in order to facilitate the process of attributing personhood to a name - useful to show the name_function works 









##OLD WORK & ORDER HERE JUST IN CASE 
#iterator to capture the first name from the name_function outputs in a non-list format 

def first_name_column_iterator (array, col_number ):
    try:
        for row in array:
            yield row[col_number]
    except IndexError:
        print ("Error, columns")
        raise

for i,j in zip(column_iterator(names2,1),column_iterator(names2,0)):
    print("First name is {}, they are a:".format(i))
    print(j)


#maybe for this function we need to do something to make the naming unique 
def name_function(df_name, names):
    list_a = []
    for x in names:
        if x is None:
            list_a.append(0)
        else: 
            nltk_results = ne_chunk(pos_tag(word_tokenize(x)))
            list_b = []
            for nltk_result in nltk_results:
                if type(nltk_result) == Tree:
                    name = ''
                    for nltk_result_leaf in nltk_result.leaves(): 
                        name += nltk_result_leaf[0]
                        list_b.append([nltk_result.label(), name])
            list_a.append(list_b)
    return list_a
    #could add the list as a column inside the function
list_A = name_function(article_df, article_df["creator"])
article_df["name_classification"] = list_A
article_df[["name_classification", "creator"]]


#Just run on author name. If a name add a new 1/0 column


sample = "John was a nice guy and had a friend named Earl"
text = article_df["creator"][0]

nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
for nltk_result in nltk_results:
    if type(nltk_result) == Tree:
        name = ''
        for nltk_result_leaf in nltk_result.leaves():
            name += nltk_result_leaf[0] + ' ' 
        print (name)
        
#above code comes from an online article. I'm thinking that I could use a pandas 
#df to figure out how to detect a name and mark the sentence Y or N or whatever. 

#YF add 
#import the Genderize package which we need to recognize names & pandas
#pip install genderize
from genderize import Genderize
import pandas as pd

#function to get the gender of the first name 
def genderize(name): 
    return Genderize().get([name])[0]["gender"]
article_df['creator gender']= article_df.apply(lambda row : genderize(row["author_first_name"]),axis=1)
print(article_df['creator gender'])
print(article_df['creator'])

#dummy df with sample names 
names = ["Shirley Temple", "Clark Gables","ashley Smith", "Alex G", "Priantha G", "John G", "john G", "jon G", "Blake Shelton", "Blake Lively", "Georgle Jungle", "George Clooney", "Danielle Duncan", "Yolanda Ferreiro"]
df_names = pd.DataFrame(names, columns = ["names"])
df_names['names']
#! Bug 2
#extract first name from a df of first and last names & create column only with first names
df_names.loc[df_names["names"].str.split().str.len() == 2, "first name"] = df_names["names"].str.split().str[0]
df_names

# apply the genderize function to all of the rows in the df_names dataset with regards to the "first name" column
df_names["gender"]= df_names.apply(lambda row : genderize(row["first name"]),axis=1)
print(df_names)


#vectors

# Here is how the people=men people did it. THey just dled this https://fasttext.cc/docs/en/english-vectors.html
#Crazy shit right. I think I'll dl and upload to the github and see if i can access it 
#though the file. Trying to be clever. 