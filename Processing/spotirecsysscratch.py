import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import json
import sys
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os



# import winsound
# duration = 1000  # milliseconds
# freq = 440  # Hz

"""#Using the Spotify Web API

App auth without user auth, user auth later. 
1. Accessing the Spotify database to create our own subset database for use
"""

client_id = "7d7875d6e08249ed8a047da889fac81f"
client_secret = "2f7cd0462cdb45a881a8bea4fd0504d8"
redirect_uri = "http://127.0.0.1:9090"
auth_manager = SpotifyClientCredentials(client_id=client_id,client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# #Authorization from Spotify

"""Copy online and copy link back after signing in"""


data = pd.read_csv('/content/drive/My Drive/SpotiRecSys/160k/data.csv')
data.head()

data.sort_values(by='popularity',ascending=False, inplace=True)

data.reset_index(drop=True, inplace=True)
data.head()

# data.drop(['id','explicit','release_date'], axis=1, inplace=True)
data.drop(['explicit','release_date'], axis=1, inplace=True)

data.head(10)

data['artists'] = data['artists'].apply(lambda x: x[1:-1].split(', ')) #string list to list
data = data.explode('artists')
data['artists'] = data['artists'].apply(lambda x: x.strip("'"))
data['artists'] = data['artists'].apply(lambda x: x.strip('\"'))

data.head()

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("paper")
sns.despine()
sns.color_palette()

# for col in data.select_dtypes(include=['float64','int64']).columns:
#   print(data[col].describe())

# fig, axes = plt.subplots(7,2,figsize=(15,25))
# for idx,col in enumerate(data.select_dtypes(include=['float64','int64']).columns):
#   sns.histplot(data[col], ax=axes[idx//2,idx%2])
#   # print(idx//2,idx%2)
# plt.show()

# print(len(data.loc[data['popularity']==0]))
print(len(data))
data.drop(data[data['popularity']<40].index, inplace=True)
len(data)
# print(len(data) - len(data.loc[data['popularity']==0]))

data.drop_duplicates(subset=["name", "artists"], keep='first');

data.drop(['key'], axis=1, inplace=True)

# plt.figure(figsize=(16, 10))
# sns.heatmap(data.corr(), annot=True)
# plt.show()

# num_data.drop(['year'], axis=1, inplace=True) 
# # Add bins: pre 20s 30s 40s 50s 60s 70s 80s 90s 00s 10s 20s

# for year in data['years']:
#   year = (year//10)*10
data['year'] = data['year'].apply(lambda x: (int(x)//10)*10)
data['year'].sample(10)


# data = pd.get_dummies(data, columns = ['mode','year'])
# data = pd.get_dummies(data, columns = ['mode'])
# data.head()

train_data = data.drop(['artists','name','popularity','duration_ms'], axis=1)
train_data.head()

num_data = train_data.select_dtypes(include=['float64', 'int64'])


num_cols = num_data.columns

cat_data = train_data.iloc[:,9:]
cat_data.head()

num_data.head()

from sklearn.cluster import KMeans, SpectralClustering, Birch, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix

minmax = MinMaxScaler()
scaled_num_data = minmax.fit_transform(num_data)

print(scaled_num_data)

scaled_num_data_df = pd.DataFrame(scaled_num_data,columns=[num_cols], index=num_data.index)
scaled_num_data_df.head()

train_data = pd.concat([scaled_num_data_df,cat_data],axis=1)
train_data.head()

# Converting from tuple to str
cols = []
for col in train_data.columns:
  if type(col)==tuple:
    cols.append(''.join(col))
  else:
    cols.append(col)
train_data.columns = cols
print(train_data.columns)
train_data.head()

"""PCA"""

from sklearn.decomposition import PCA
pca = PCA(2)
principal_components = pca.fit_transform(scaled_num_data)

principal_df = pd.DataFrame(data=principal_components, columns=['PC1','PC2'])
principal_df

# plt.figure(figsize=(30,30))
# sns.scatterplot(x = principal_df['PC1'], y = principal_df['PC2'])
# plt.show()

"""Elbow Method"""

# sse = []
# list_k = list(range(1,100))

# start_time = time.time()
# for k in list_k:
#   print(k)
#   kmeans = MiniBatchKMeans(n_clusters=k)
#   kmeans.fit(scaled_num_data)
#   sse.append(kmeans.inertia_)
#   print(time.time()-start_time)
# # winsound.Beep(freq, duration)

# sse_df = pd.DataFrame(sse, columns=['sse'])
# sse_df.head()
# # sse_df.to_csv('/content/drive/My Drive/SpotiRecSys/sse_30nc_mini.csv')

# plt.figure(figsize = (15,15))
# plt.plot(list_k, sse, '-o')
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance')
# plt.show()
# plt.savefig('/content/drive/My Drive/SpotiRecSys/elbow_160k_30nc.png')

# from scipy.cluster.hierarchy import dendrogram, linkage
# plt.figure(figsize=(10,7))
# dendrogram(linkage(num_data.sample(1000), 'average'), show_leaf_counts=True)
# plt.show()

from sklearn import preprocessing
normalized_vectors = preprocessing.normalize(scaled_num_data_df)

# pd.DataFrame(normalized_vectors, columns=num_data.columns).describe()

# kmeans = KMeans(n_clusters = 11, max_iter=100, random_state = 42)
# kmeans.fit(normalized_vectors)
# kmeans.inertia_

# import pickle
# pickle.dump(kmeans, open("/content/drive/My Drive/SpotiRecSys/save.pkl", "wb"))

import pickle
kmeans = pickle.load(open("/content/drive/My Drive/SpotiRecSys/save.pkl", "rb"))

# silhouette_score(normalized_vectors, kmeans_cosine.labels_, metric = 'euclidean')

# kmeans = KMeans(n_clusters=12, max_iter=100)
# kmeans.fit(scaled_num_data)
# kmeans.inertia_

# silhouette_score(scaled_num_data, kmeans.labels_, metric='cosine')

# y_pred = kmeans.predict(scaled_num_data_df)

labels = []
labels = kmeans.labels_

data['cluster'] = labels
data.head(50)

# VISUALISATION
data['constant'] = 'data'
# for col in data.columns:
sns.stripplot(x=data['constant'], y=data['acousticness'],hue=data['cluster'], jitter=True, dodge=True)
plt.show()

"""VISUALISATIONS"""

# f, axes = plt.subplots(4, 3, figsize=(20, 55), sharex=False) 

# f.subplots_adjust(hspace=0.2, wspace=0.7) 


# # for i in range(len(list(num_data))): 
# for i, col in enumerate(data.select_dtypes(include=['float64', 'int64'])):
#     # col = num_data.columns[i]
#     ax = sns.stripplot(x=data['constant'],y=data[col].values,hue=data['cluster'],jitter=True,ax=axes[i//3,i%3], size=7, dodge=True)
#     ax.set_title(col)
# # plt.savefig('/content/drive/My Drive/SpotiRecSys/kmeans_nc9.png') # Change graph saving setting

# sns.clustermap(num_data, figsize = (50,50))
# plt.show()

# principal_df = pd.concat([principal_df, data['cluster']],axis=1)
# principal_df.head()

# plt.figure(figsize=(30,30))
# sns.scatterplot(x = principal_df['PC1'], y = principal_df['PC2'])
# plt.show()

data.loc[data['cluster']==8].head(30)

# data.loc[data['cluster']==13].sample(30) #Metal
# data.loc[data['cluster']==11].sample(30) # Hip Hop
data.loc[data['cluster']==6].sample(30)
# data.loc[data['cluster']==16].sample(30)

# for i in range(12):
#   print(data.loc[data['cluster']==i].mean())

with open('/content/drive/My Drive/SpotiRecSys/160k/super_genres.json') as f:
  super = json.load(f)

scores = []
for i, s in enumerate(super):
  scores.append(s['score'])
print(scores)
sns.lineplot(data = scores, x = range(40), y=scores, marker='o')

i=8
super[i]['score'], super[i]['n_clusters']

# super[9]['predictions']

"""Prediction"""

search_song = 'Fake Plastic Trees'

search_artists = '' # Will be input() blank in final
# search_artists = 'Years & Years'

search = search_song + ' ' + search_artists
search

# data.reset_index().index[2] # iloc indexing and reset indexing

print(sp.search(q=search, type='track,artist')['tracks']['items'][0]['album']['artists'][0]['name'])
search_id = sp.search(q=search, type='track')['tracks']['items'][0]['id']
search_id

try:
  search_index = data[data['id']==search_id].index.to_list()[0]
except ValueError:
# else:
  search_features = pd.DataFrame(sp.audio_features(search_id)[0], index=[0])
# search_index = data[data['id']=='asdsad'].index.to_list()[0]
search_index # pd reset index
# data['index'].iloc[search_index] # original indexing including repeats

# # Need to run this if song not found in db 
# # TODO: Add normalization(transform)
# del_features = ['analysis_url','duration_ms','id','key','mode','time_signature','track_href','type','uri']

# search_features = ((sp.audio_features(search_id)[0]))

# for delt in del_features:
#   del search_features[delt]
# # search_features = np.squeeze(np.array(pd.DataFrame(search_features, index=[0])))
# index_last = num_data.index[-1]
# index_last
# search_features = pd.DataFrame(search_features, index=[index_last+1])
# search_features

# num_data = pd.concat([num_data, search_features], axis=0)
# preprocessing.normalize(num_data)

data[data.index==search_index]

search_features = np.squeeze(num_data[data.index == search_index].iloc[0].values)
search_features

# num_data[data.index==2].values

from scipy.spatial import distance

feature_distances = []

start_time = time.time()
for i in num_data.index:
  
  song_feature = num_data[num_data.index == i].iloc[0].values
  curr_distance = distance.cosine(search_features, song_feature)
  feature_distances.append(curr_distance)
  if curr_distance == 0.0:
    print(i)
time.time() - start_time

rec_indices = np.argsort(feature_distances)[:20].tolist()
rec_indices

np.sort(feature_distances)[:20].tolist()

# rec1 = num_data.iloc[(np.argsort(feature_distances)[0])].values
# rec1

# (np.argsort(feature_distances)[0])

# num_data.iloc[2852].values

# num_data[num_data.index == 1883].values

# num_data[num_data.index == 2852].values

# distance.cosine(search_features, rec1)

rec_songs = []

for index in rec_indices:
  rec_songs.append(data.iloc[index])
# rec_songs

ind = 1

rec_indices[ind]

distance.cosine(search_features, num_data.iloc[rec_indices[ind]].values)

pd.DataFrame(data.iloc[rec_indices[ind]]).T

# Song searched for df
data[data.index==search_index].head()

num_data[num_data.index == rec_indices[ind]]

"""Computing cosine similarity matrix to decrease computational time considerably"""

# # Using cosine_similarity

# from sklearn.metrics.pairwise import cosine_similarity

# sub = num_data.values[:30000, :]

# cosine_sim = cosine_similarity(sub, sub)

# np.save("/content/drive/My Drive/SpotiRecSys/sim.npy", cosine_sim)

# song_id["search"] = song_id["name"] + ' ' + song_id["artists"]
# song_id.to_csv("id2.csv", index=False)

# def get_recommendations(idx, cosine_sim):
#     sim_scores = list(enumerate(cosine_sim[idx]))

#     # Sort the songs based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the scores of the 10 most similar songs
#     sim_scores = sim_scores[1:11]
#     sim_scores.sort()
#     # Get the song indices
#     song_index = [i[0] for i in sim_scores]
#     song_index = song_index
    
#     # Return the top 10 most similar songs
#     return song_index

# search_features = np.squeeze(num_data[data.index == search_index].iloc[0].values)

# for i in num_data.index:
  
#   song_feature = num_data[num_data.index == i].iloc[0].values
#   curr_distance = distance.cosine(search_features, song_feature)
#   feature_distances.append(curr_distance)

# def get_recommendations(song_index, cosine_sim):
#     sim_scores = list(enumerate(cosine_sim[song_index])) # Get the pairwise scores of the requested song as a list

#     # Sort the obtained scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the top 10 recs
#     sim_scores = sim_scores[1:11]
#     sim_scores.sort()

#     print(sim_scores)
#     # Get the song indices
#     song_index = [i[0] for i in sim_scores]
#     return song_index

# sim = np.load("/content/drive/My Drive/SpotiRecSys/sim.npy")

# get_recommendations(song_index, sim)

# sim = np.load("/content/drive/My Drive/SpotiRecSys/sim.npy")
# all_res = np.zeros((1, 10))

# for i in range(0, 30000):
#     rec_list = get_recommendations(i, sim)
#     rec_array = np.array(rec_list)
#     rec_array = rec_array.reshape((1, 10))
#     rec_array = rec_array.astype(int)
#     all_res = np.append(all_res, rec_array, axis=0)
#     print("saved for:" + str(i))

# all_res = np.delete(all_res, 0, axis=0)
# np.save("light.npy", all_res)
# all_res

# def generate_recoms(idx):
#     """
#     fetches recommendations for given song (id)
#     """
#     idx = int(idx)

#     # light.npy is pre-saved recommendations for all songs to optimize time
#     sim = np.load("data/light.npy")
    
#     df = pd.read_csv("data/id2.csv")
#     recoms_list = sim[idx, :]
#     recommendation = []
#     for i in range(0, 10):
#         song_name = df.iloc[int(recoms_list[i]), 1]
#         artist_name = df.iloc[int(recoms_list[i]), 2]
#         spotify_id = df.iloc[int(recoms_list[i]), 3]
#         image_url, prev_url = spotify_api.get_urls(str(spotify_id))
#         temp_dict = {
#             "song_name": song_name,
#             "artist_name": artist_name,
#             "spotify_id": spotify_id,
#             "image_url": image_url,
#             "preview": prev_url
#         }
#         recommendation.append(temp_dict)
#     return recommendation

# pd.DataFrame(all_res)

# all_res[search_index]

# data[data.index==search_index]

# Prediction as df



# from sklearn.metrics.pairwise import cosine_similarity

# batch_size = 1000

# assert num_data.shape[1] == num_data.shape[1]
# ret = np.ndarray((num_data.shape[0], num_data.shape[0]))
# for row_i in range(0, int(num_data.shape[0] / batch_size) + 1):
#     start = row_i * batch_size
#     end = min([(row_i + 1) * batch_size, num_data.shape[0]])
#     if end <= start:
#         break 
#     rows = num_data[start: end]
#     sim = cosine_similarity(rows, num_data) # rows is O(1) size
#     ret[start: end] = sim

# # Using linear_kernel

# from sklearn.metrics.pairwise import linear_kernel, pairwise_kernels

# cosine_sim = pairwise_kernels(num_data, num_data, metrix='cosine', n_jobs = )


# # cosine_sim = linear_kernel(num_data, num_data)



"""Using Euclidean Distances, Pearson Correlation"""

e_feature_distances = []

start_time = time.time()
for i in num_data.index:
  
  song_feature = num_data[num_data.index == i].iloc[0].values
  curr_distance = np.linalg.norm(search_features - song_feature)
  e_feature_distances.append(curr_distance)
  if curr_distance == 0.0:
    print(i)
time.time() - start_time

erec_indices = np.argsort(e_feature_distances)[:20].tolist()
erec_indices

np.sort(e_feature_distances)[:20].tolist()

erec_songs = []

for index in erec_indices:
  erec_songs.append(data.iloc[index])
for i in erec_songs:
  print(i['name'])

ind = 1
erec_indices[ind]

pd.DataFrame(data.iloc[erec_indices[ind]]).T

data[data.index==search_index].head()

num_data[num_data.index==search_index].head()

"""PCA"""

pca = PCA()
principalComponents = pca.fit_transform(num_data)
principalComponents

print(sum(pca.explained_variance_ratio_))
print(sum(pca.explained_variance_ratio_[:3]))

pca_data = pd.DataFrame(pca.transform(num_data))
pca_data = pca_data.iloc[:,:3]
pca_data

pca_feature_distances = []

pca_search_feature = pca_data[pca_data.index==search_index]

pca_search_feature

from scipy.spatial import distance

start_time = time.time()
for i in pca_data.index:
  
  pca_song_feature = pca_data[pca_data.index == i].values
  curr_distance = distance.cosine(np.array(pca_search_feature), pca_song_feature)
  pca_feature_distances.append(curr_distance)
  if curr_distance == 0.0:
    print(i)
time.time() - start_time

pcrec_indices = np.argsort(pca_feature_distances)[:20].tolist()
pcrec_indices

np.sort(pca_feature_distances)[:20].tolist()

pcrec_songs = []

for index in pcrec_indices:
  pcrec_songs.append(data.iloc[index])
for i in pcrec_songs:
  print(i['name'])

"""Analysing User Library"""

# user = pd.read_csv("/content/drive/My Drive/SpotiRecSys/User_10k_songs.csv")
# user.head(15)

# with open("/content/drive/My Drive/SpotiRecSys/keys2046.json", "w") as outfile:  
#     json.dump(keys_dict, outfile)

# #SAVING FEATURE FILE FROM JSON TO DICT TO DF TO CSV
# with open("/content/drive/My Drive/SpotiRecSys/keys2046.json") as f:
#     keys_json = json.load(f)
# print(keys_json)
# keys_2084 = pd.DataFrame.from_dict(keys_json)
# # keys_2084.head(15)
# keys_2084.to_csv(f'/content/drive/My Drive/SpotiRecSys/2084features.csv')

# user_features = pd.read_csv('/content/drive/My Drive/SpotiRecSys/library_features.csv')
# user_features = user_features.iloc[:,1:]
# user_features.head(10)



"""# With Genre"""

data.head()

data_genres = pd.read_csv("/content/drive/My Drive/SpotiRecSys/160k/data_w_genres.csv")

del_cols = data.iloc[:,1:14].columns
list(del_cols)

data_genres.drop(data_genres.columns[1:14],axis=1, inplace=True)
data_genres.head(10)

data_genres.sort_values('count', ascending=False)

data_comb = data.reset_index().merge(data_genres, how="inner", on=['artists']).set_index('index')
data_comb.head()

# idx = 0

# try:
#   data['artists'] = data['artists'].apply(lambda x: x[1:-1].split(', ')) #string list to list

# except ValueError:
#   idx = idx + 1

# print(idx)
# # idx = 0
# # idxs = []
# # for genre in data_comb['genres']:
# #   # print(type(genre))

   

# #   try:
# #     genre = genre[-1:1].split(', ')

# #   except TypeError:
# #     if type(genre) == float:
# #       idxs.append(idx) 


# #   idx = idx + 1

# # idxs
data_comb['genres'][0]

# data_comb = data_comb[data_comb.index==0]['genres'][0]).strip('][').split(', ')

for genre in data_comb['genres']:

  try:
    genre = genre.strip('][').split(', ')
  except AttributeError:
    genre = []
    idxs.append(idx)

  idx = idx + 1


data_comb['genres']

data_comb.iloc[idxs]

len(idxs)

data_comb.shape

data_comb.drop(data_comb.index[idxs], axis=0, inplace=True)

# Checking to see if all genres are now strings

for genre in data_comb['genres']:
  if type(genre)==float:
    print(genre)

data_comb['genres'] = data_comb['genres'].apply(lambda x: x.split(', ').strip("'"))

data_comb['genres'][0][0]



data = pd.read_csv('/content/drive/My Drive/SpotiRecSys/160k/data.csv')
data.sort_values(['popularity'], ascending=False, inplace=True)
data.reset_index(drop=True, inplace=True)
data