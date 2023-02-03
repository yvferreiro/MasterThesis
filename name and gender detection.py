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

#need to clearly use more research. lol