# https://www.kaggle.com/tmdb/tmdb-movie-metadata  veriler buradan alınacak
import difflib
import pandas as pd
import matplotlib.pyplot as plt
import json

from prompt_toolkit.key_binding.bindings.named_commands import uppercase_word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

df = pd.read_csv('FilmData.txt') # read the data
#print(df.head())   #whats inside the data frame

x = df.iloc[0]
#print(x)

#print(x['genres'])
#print(x['keywords'])

j=json.loads(x['genres'])
#print(j)

filmFull = ' '.join(''.join(jj['name'].split()) for jj in j)

#print(filmFull)

def genres_and_keywords_to_string(row):
    genres = json.loads(row['genres'])
    genres = ' '.join(''.join(j['name'].split()) for j in genres)

    keywords = json.loads(row['keywords'])
    keywords = ' '.join(''.join(j['name'].split()) for j in keywords)

    return "%s %s" % (genres, keywords)

df['string'] = df.apply(genres_and_keywords_to_string, axis=1)

# creating a tf-idf vectorizer
tfidf = TfidfVectorizer(max_features=2000)

#create a data matrix from the overiew
x = tfidf.fit_transform(df['string'])

# print(x)

# generate a mapping from movie title

movie2idx = pd.Series(df.index ,index=df['title'].str.lower())

#print(movie2idx)

idx = movie2idx['Scream 3'.lower()]
#print(idx)

query = x[idx]
#print(query)

#print(query.toarray())

#cosine similarity between query and every vector in x

scores = cosine_similarity(query,x)
#print(scores)

scores = scores.flatten()

plt.plot(scores)

(-scores).argsort()

plt.plot(scores[(-scores).argsort()])
recommended_idx= (-scores).argsort()[1:6]


#print(df['title'].iloc[recommended_idx])

# create a function that generates recommendations
def recommend(title):
  # get the row in the dataframe for this movie
  title = title.strip().lower()
  if title not in movie2idx:
    prob = difflib.get_close_matches(title, movie2idx.index, n=3, cutoff=0.6)
    return f"'{title}' named film not found.\nDid you mean this?\n- " + '\n- '.join(prob)

  idx = movie2idx[title.lower()]
  if type(idx) == pd.Series:
    idx = idx.iloc[0]

  # calculate the pairwise similarities for this movie
  query = x[idx]
  scores = cosine_similarity(query, x)

  # currently the array is 1 x N, make it just a 1-D array
  scores = scores.flatten()

  # get the indexes of the highest scoring movies
  # get the first K recommendations
  # don't return itself!
  recommended_idx = (-scores).argsort()[1:6]

  # return the titles of the recommendations
  return df['title'].iloc[recommended_idx]


#print(recommend('The Calling'))    # bu kısmı input alacak şekilde yaparsan tamamen kullanıcı odaklı bir
                                    # film önerme projesi olmuş olur

print(recommend(input("What film do you want to watch?")))



