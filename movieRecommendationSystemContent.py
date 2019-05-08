import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')

credits = pd.read_csv('credits.csv')
credits_foreign = pd.read_csv('credits_foreign_movies.csv')
keywords = pd.read_csv('keywords.csv')
keywords_foreign = pd.read_csv('keywords_foreign_movies.csv')
links_small = pd.read_csv('links_small.csv')
links_foreign = pd.read_csv('links_foreign_movies.csv')
md = pd.read_csv('movies_metadata.csv')
md_foreign = pd.read_csv('foreignfilms.csv')
ratings = pd.read_csv('ratings_small.csv')
ratings_foreign = pd.read_csv('ratings_foreign_movies.csv')


print(credits.head())

credits.columns
credits.shape
print(credits.info())

print(credits_foreign.head())

credits_foreign.columns
credits_foreign.shape
print(credits_foreign.info())

print(keywords.head())

keywords.columns
keywords.shape

print(keywords.info())

print(keywords_foreign.head())

keywords_foreign.columns
keywords_foreign.shape

print(keywords_foreign.info())

print(links_small.head())

links_small.columns
links_small.shape

print(links_small.info())

print(links_foreign.head())

links_foreign.columns
links_foreign.shape

print(links_foreign.info())

print(md.iloc[0:3].transpose())

md.columns
md.shape

print(md.info())

print(md_foreign.iloc[0:3].transpose())

md_foreign.columns
md_foreign.shape

print(md_foreign.info())

print(ratings.head())

ratings.columns
ratings.shape

print(ratings.info())

print(ratings_foreign.head())

ratings_foreign.columns
ratings_foreign.shape

print(ratings_foreign.info())

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i[
    'name'] for i in x] if isinstance(x, list) else [])

md_foreign['genres'] = md_foreign['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i[
    'name'] for i in x] if isinstance(x, list) else [])

vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')

vote_counts_foreign = md_foreign[md_foreign['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages_foreign = md_foreign[md_foreign['vote_average'].notnull()]['vote_average'].astype('int')

C = vote_averages.mean()
print(C)

C_foreign = vote_averages_foreign.mean()
print(C_foreign)

m = vote_counts.quantile(0.95)
print(m)

m_foreign = vote_counts_foreign.quantile(0.55)
print(m_foreign)

md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

md_foreign['year'] = pd.to_datetime(md_foreign['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

qualified = md[(md['vote_count'] >= m) &
               (md['vote_count'].notnull()) &
               (md['vote_average'].notnull())][['title',
                                                'year',
                                                'vote_count',
                                                'vote_average',
                                                'popularity',
                                                'genres']]

qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')

qualified_foreign = md_foreign[(md_foreign['vote_count'] >= m_foreign) &
               (md_foreign['vote_count'].notnull()) &
               (md_foreign['vote_average'].notnull())][['title',
                                                'year',
                                                'vote_count',
                                                'vote_average',
                                                'popularity',
                                                'genres']]

qualified_foreign['vote_count'] = qualified_foreign['vote_count'].astype('int')
qualified_foreign['vote_average'] = qualified_foreign['vote_average'].astype('int')

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)

qualified_foreign['wr'] = qualified_foreign.apply(weighted_rating, axis=1)
qualified_foreign = qualified_foreign.sort_values('wr', ascending=False).head(250)

print(qualified.head(15))
print(qualified_foreign.head(15))

s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)
print(gen_md.head(3).transpose())

s_foreign = md_foreign.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s_foreign.name = 'genre'
gen_md_foreign = md_foreign.drop('genres', axis=1).join(s)
print(gen_md_foreign.head(3).transpose())

def build_chart(genre, percentile=0.85):
    df = gen_md_foreign[gen_md_foreign['genre'] == genre]
    vote_counts_foreign = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages_foreign = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C_foreign = vote_averages_foreign.mean()
    m_foreign = vote_counts_foreign.quantile(percentile)

    qualified_foreign = df[(df['vote_count'] >= m_foreign) & (df['vote_count'].notnull()) &
                   (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genre']]
    qualified_foreign['vote_count'] = qualified_foreign['vote_count'].astype('int')
    qualified_foreign['vote_average'] = qualified_foreign['vote_average'].astype('int')

    qualified_foreign['wr'] = qualified_foreign.apply(lambda x:
                        (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C),
                        axis=1)
    qualified_foreign = qualified_foreign.sort_values('wr', ascending=False).head(250)

    return qualified_foreign

print(build_chart('Romance').head(15))

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

links_foreign = links_foreign[links_foreign['tmdbId'].notnull()]['tmdbId'].astype('int')

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

md['id'] = md['id'].apply(convert_int)
print(md[md['id'].isnull()])

md_foreign['id'] = md_foreign['id'].apply(convert_int)
print(md_foreign[md_foreign['id'].isnull()])

md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
md_foreign['id'] = md_foreign['id'].astype('int')

smd = md[md['id'].isin(links_small)]
print(smd.shape)

smd_foreign = md_foreign[md_foreign['id'].isin(links_foreign)]
print(smd_foreign.shape)

smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

smd_foreign['tagline'] = smd_foreign['tagline'].fillna('')
smd_foreign['description'] = smd_foreign['overview'] + smd_foreign['tagline']
smd_foreign['description'] = smd_foreign['description'].fillna('')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

tf_foreign = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_foreign = tf.fit_transform(smd_foreign['description'])

print(tfidf_matrix.shape)
print(tfidf_matrix_foreign.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim[0])

cosine_sim_foreign = linear_kernel(tfidf_matrix_foreign, tfidf_matrix_foreign)
print(cosine_sim_foreign[0])

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

smd_foreign = smd_foreign.reset_index()
titles_foreign = smd_foreign['title']
indices_foreign = pd.Series(smd_foreign.index, index=smd_foreign['title'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_foreign[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:3]
    movie_indices = [i[0] for i in sim_scores]
    return titles_foreign.iloc[movie_indices]

print(get_recommendations('Small Faces').head(10))

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')
print(md.shape)

keywords_foreign['id'] = keywords_foreign['id'].astype('int')
credits_foreign['id'] = credits_foreign['id'].astype('int')
md_foreign['id'] = md_foreign['id'].astype('int')
print(md_foreign.shape)

md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')

md_foreign = md_foreign.merge(credits, on='id')
md_foreign = md_foreign.merge(keywords, on='id')

smd = md[md['id'].isin(links_small)]
print(smd.shape)

smd_foreign = md_foreign[md_foreign['id'].isin(links_foreign)]
print(smd_foreign.shape)

smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

smd_foreign['cast'] = smd_foreign['cast'].apply(literal_eval)
smd_foreign['crew'] = smd_foreign['crew'].apply(literal_eval)
smd_foreign['keywords'] = smd_foreign['keywords'].apply(literal_eval)
smd_foreign['cast_size'] = smd_foreign['cast'].apply(lambda x: len(x))
smd_foreign['crew_size'] = smd_foreign['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x, x])

smd_foreign['director'] = smd_foreign['crew'].apply(get_director)
smd_foreign['cast'] = smd_foreign['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd_foreign['cast'] = smd_foreign['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd_foreign['keywords'] = smd_foreign['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd_foreign['cast'] = smd_foreign['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd_foreign['director'] = smd_foreign['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd_foreign['director'] = smd_foreign['director'].apply(lambda x: [x,x, x])

s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
print(s[:5])

s_foreign = smd_foreign.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s_foreign.name = 'keyword'
s_foreign = s_foreign.value_counts()
print(s_foreign[:5])

s = s[s > 1]

s_foreign = s_foreign[s_foreign > 1]

stemmer = SnowballStemmer('english')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


smd_foreign['soup'] = smd_foreign['keywords'] + smd_foreign['cast'] + smd_foreign['director'] + smd_foreign['genres']
smd_foreign['soup'] = smd_foreign['soup'].apply(lambda x: ' '.join(x))

smd_foreign['keywords'] = smd_foreign['keywords'].apply(filter_keywords)
smd_foreign['keywords'] = smd_foreign['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd_foreign['keywords'] = smd_foreign['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

count_foreign = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix_foreign = count.fit_transform(smd_foreign['soup'])

cosine_sim_foreign = cosine_similarity(count_matrix_foreign, count_matrix_foreign)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

smd_foreign = smd_foreign.reset_index()
titles_foreign = smd_foreign['title']
indices_foreign = pd.Series(smd.index, index=smd['title'])

def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_foreign[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd_foreign.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) &
                       (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified

print(improved_recommendations('Small Faces'))
