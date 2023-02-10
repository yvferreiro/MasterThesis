import io

#good god. I gave this almost 25 minutes to load and nothing has happened. I'll
#read some documentation and come back. 

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

file = "C:/Users/danie/Desktop/crawl-300d-2M-subword.vec"

data = load_vectors(file)

print("hi")
