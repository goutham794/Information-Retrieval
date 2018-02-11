
# coding: utf-8

# In[8]:

import re
import string


# ## Reading in the data.
# "texts" is a list of all the articles.
# "urls" is a list of all the corresponding urls.
# "processed_corpus" is a list of list which contains all tokens in lower case without stop words.

# In[2]:

doc = open("./data/nytimes_sample.txt").read()
texts = []
urls = []
article = ""
for line in doc.splitlines():
    if re.findall("URL.*", line) != []:
        urls.append(line)
        texts.append(article)
        article = ""

    else:
        article += line
del texts[0]

stopwords = set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by',
    'for', 'if', 'in', 'into', 'is', 'it',
    'no', 'not', 'of', 'on', 'or', 's', 'such',
    't', 'that', 'the', 'their', 'then', 'there', 'these',
    'they', 'this', 'to', 'was', 'will', 'with'
])


processed_corpus = [[word for word in text.lower().split() if word.isalnum() and word not in stopwords]for text in texts]


# ## Accepting query from user

# In[9]:

string_corpus = [" ".join(article) for article in processed_corpus]

query = input("enter query")

string_corpus.append(query)


# ## TfIdf matrix is made for all the documents and query.
# ### Cosine similarities are calculated for all documents with the query which are then sorted. SInce we want descending order of cosine similarity values, we multiply with -1. 
# 

# In[13]:

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(string_corpus)

from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix).flatten()

import numpy as np
ranked_indices = np.argsort(-1 * cosine_similarities)
    


# ### The first element of ranked_indices is removed since it is the cosine similarity of the query with itself.
# ### The first ten inidces of the descending order of cosine similarities are taken in a list and correspoding urls of the articles are printed out

# In[12]:

ranked_indices = np.delete(ranked_indices, 0)
rank = list(ranked_indices)[0:10]

for n in rank:
    print (urls[n])

