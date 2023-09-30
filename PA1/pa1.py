{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['yugoslav', 'author', 'plan', 'arrest', 'eleven', 'coal', 'miner', 'opposit', 'politician', 'suspicion', 'sabotag', 'connect', 'strike', 'action', 'presid', 'slobodan', 'milosev', 'listen', 'bbc', 'news', 'world']\n"
     ]
    }
   ],
   "source": [
    "from nltk.stem import PorterStemmer\n",
    "\n",
    "# Load the stopwords from a file\n",
    "# https://gist.github.com/larsyencken/1440509\n",
    "with open('stopwords.txt', 'r') as f:\n",
    "    stop_words = set(f.read().splitlines())\n",
    "\n",
    "# Load the input text\n",
    "text = \"\"\"And Yugoslav authorities are planning the arrest of eleven coal miners \n",
    "and two opposition politicians on suspicion of sabotage, that's in \n",
    "connection with strike action against President Slobodan Milosevic. \n",
    "You are listening to BBC news for The World.\"\"\"\n",
    "\n",
    "# Step 1: Tokenization using basic Python functions (without external library)\n",
    "# Remove punctuations and split the text into words\n",
    "translator = str.maketrans('', '', '''!\"#$%&'()*+,-./:;<=>?@[\\\\]^_`{|}~''')\n",
    "tokens = text.translate(translator).split()\n",
    "\n",
    "# Step 2: Lowercasing\n",
    "tokens = [word.lower() for word in tokens]\n",
    "\n",
    "# Step 3: Stemming using Porter’s algorithm\n",
    "ps = PorterStemmer()\n",
    "tokens = [ps.stem(word) for word in tokens]\n",
    "\n",
    "# Step 4: Stopword Removal using the loaded stopwords set\n",
    "tokens = [word for word in tokens if word not in stop_words]\n",
    "\n",
    "# Step 5: Save the result as a txt file\n",
    "with open('result.txt', 'w') as f:\n",
    "    for token in tokens:\n",
    "        f.write(\"%s\\n\" % token)\n",
    "\n",
    "# Print out the tokens for verification\n",
    "print(tokens)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
