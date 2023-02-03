import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

sample = "John was a nice guy and had a friend named Earl"
text = '''
This is a sample text that contains the name Alex Smith who is one of the developers of this project.
You can also find the surname Jones here.
'''

nltk_results = ne_chunk(pos_tag(word_tokenize(sample)))
for nltk_result in nltk_results:
    if type(nltk_result) == Tree:
        name = ''
        for nltk_result_leaf in nltk_result.leaves():
            name += nltk_result_leaf[0] + ' '
        print ('Type: ', nltk_result.label(), 'Name: ', name)
        
#above code comes from an online article. I'm thinking that I could use a pandas 
#df to figure out how to detect a name and mark the sentence Y or N or whatever. 

#YF add 
#import the Genderize package which we need to recognize names & pandas
#pip install genderize
from genderize import Genderize
import pandas as pd

#function to detect the gender 
def genderize(name): 
    return Genderize().get([name])[0]["gender"]

#dummy df with sample names 
names = ["Shirley Temple", "Clark Gables", "Georgle Jungle", "George Clooney", "Danielle Duncan", "Yolanda Ferreiro"]
df_names = pd.DataFrame(names, columns = ["names"])

#extract first name from a df of first and last names & create column only with first names
df_names.loc[df_names["names"].str.split().str.len() == 2, "first name"] = df_names["names"].str.split().str[0]
df_names

# apply the genderize function to all of the rows in the df_names dataset with regards to the "first name" column
df_names["gender"]= df_names.apply(lambda row : genderize(row["first name"]),axis=1)
print(df_names)
