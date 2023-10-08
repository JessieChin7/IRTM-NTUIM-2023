from nltk.stem import PorterStemmer

# Load the stopwords from a file
# https://gist.github.com/larsyencken/1440509
with open('stopwords.txt', 'r') as f:
    stop_words = set(f.read().splitlines())

# Load the input text
text = """And Yugoslav authorities are planning the arrest of eleven coal miners 
and two opposition politicians on suspicion of sabotage, that's in 
connection with strike action against President Slobodan Milosevic. 
You are listening to BBC news for The World."""

# Step 1: Tokenization using basic Python functions (without external library)
# Remove punctuations and split the text into words
translator = str.maketrans('', '', '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~''')
tokens = text.translate(translator).split()

# Step 2: Lowercasing
tokens = [word.lower() for word in tokens]

# Step 3: Stopword Removal using the loaded stopwords set
tokens = [word for word in tokens if word not in stop_words]

# Step 4: Stemming using Porter’s algorithm
ps = PorterStemmer()
tokens = [ps.stem(word) for word in tokens]

# Step 5: Save the result as a txt file
with open('result.txt', 'w') as f:
    for token in tokens:
        f.write("%s\n" % token)

# Print out the tokens for verification
print(tokens)
