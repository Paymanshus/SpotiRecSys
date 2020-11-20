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
import joblib
import time
import os
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

import auth
# from PlaylistSearch import playlist_search

pd.set_option('display.width', 500)
np.set_printoptions(linewidth=500)
pd.set_option('display.max_columns', 30)

'''
Read processed data and num_data (with labels)
Take song input from user (+artist name) and get song_id for it
Search for song_id in data
    If song in data: find song in data, load search_features
    If song not in data: sp.search to query for song, get search_features, normalize
Compute distances of search_song with each song in data
Give user a list of 10 best recs
'''


def get_search(sp, search_song, search_artists=''):
    search_song = "No surprises"
    search = search_song + ' ' + search_artists

    print("Artist selected: " + sp.search(q=search, type='track,artist')['tracks']['items'][0]['album']['artists'][0][
        'name'])
    search_id = sp.search(q=search, type='track')['tracks']['items'][0]['id']
    print(search_id)

    search_index = pdata[pdata['id'] == search_id].index.tolist()[0]

    print(search_index)
    print(pdata[pdata.index == 3325])
    print(scaled_num_data[pdata.index == search_index])
    input('index check')
    # try:
    #     search_index = pdata[pdata['id'] == search_id].index.tolist()[0]
    # except ValueError:
    #     # else:
    #     search_features = pd.DataFrame(sp.audio_features(search_id)[0], index=[0])
    '''
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

    # Normalization of search features if not found in dataset 
    # minmax_scaler = joblib.load('/Processing/minmax_scaler.save')
    search_features = minmax_scaler.transform(search_features)
    normalizer_scaler = joblib.load('/Processing/normalizer_scaler_save')
    search_features = normalizer_scaler.transform(search_features)

    data[data.index == search_index]
    '''
    search_features = np.squeeze(
        scaled_num_data[pdata.index == search_index].iloc[0].values)

    return search_index, search_features


def predict(search_index, search_features):

    # Cosine Recommendation Model

    feature_distances = []

    feature_distances = cosine_similarity(
        scaled_num_data, search_features.reshape(1, -1))

    print(pd.DataFrame(feature_distances))
    input('Continue')

    # Displaying song recommendations

    # Remove from df conversion till here if not needed to print
    feature_distances = np.squeeze(feature_distances)
    rec_indices = np.argsort(feature_distances)[-25:-1].tolist()[::-1]
    print(f'List of indices of recommendations are: {rec_indices}')

    rec_songs = []

    for index in rec_indices:
        rec_songs.append(pdata.iloc[index])

    prev = 0

    for idx, song in enumerate(rec_songs):
        if song['name'] == prev:  # If current song is the same as the last one
            rec_songs.pop(idx)
            rec_indices.pop(idx)
        else:
            print(idx, song['name'], ' -', song['artists'])
            prev = song['name']

    input('Continue')
    # Comparison of songs with selected song

    ind = 0

    # Recommended song's features
    print("Top recommended song\'s features: ")
    print(pd.DataFrame(pdata.iloc[rec_indices[ind]]).T)

    # Song searched for df
    print('Original song features')
    print(pdata[pdata.index == search_index].head())
    input('Continue')

    # EUCLIDEAN DISTANCE MODEL

    e_feature_distances = []

    e_feature_distances = np.linalg.norm(
        scaled_num_data.sub(np.array(search_features)), axis=1)
    pd.DataFrame(e_feature_distances)

    erec_indices = np.argsort(e_feature_distances)[1:20].tolist()
    print(f'List of indices of recommendations are: {erec_indices}')

    print(f'Distances are given as: ')
    print(np.sort(feature_distances)[:20].tolist())

    erec_songs = []

    for index in erec_indices:
        erec_songs.append(pdata.iloc[index])

    prev = 0

    for idx, song in enumerate(erec_songs):
        if song['name'] == prev:  # If current song is the same as the last one
            erec_songs.pop(idx)
        else:
            print(idx, song['name'], ' -', song['artists'])
            prev = song['name']


def artist_rec(search_features):
    data_artists.drop(
        data_artists[data_artists['count'] < 50].index, inplace=True)
    data_artists.drop(['key', 'count', 'mode', 'duration_ms',
                       'popularity'], axis=1, inplace=True)
    num_data_artists = data_artists.select_dtypes(include=['float64', 'int64'])

    minmax = joblib.load("Processing/minmax_scaler.save")
    scaled_num_data_artists_df = pd.DataFrame(
        minmax.fit_transform(num_data_artists))

    # Cosine Similarity Based Artist Recommendation
    artist_feature_distances = np.squeeze(cosine_similarity(
        scaled_num_data_artists_df, search_features.reshape(1, -1)))
    artist_rec_indices = np.argsort(
        artist_feature_distances)[-21:].tolist()[::-1]
    # np.sort(artist_feature_distances)[-21:].tolist()[::-1]
    for artist in artist_rec_indices:
        print(data_artists.iloc[artist]['artists'])

    input('Continue')

    # Euclidean Based

    artist_feature_distances = np.linalg.norm(
        scaled_num_data_artists_df.sub(np.array(search_features)), axis=1)
    artist_rec_indices = np.argsort(artist_feature_distances)[:25].tolist()

    rec_artists = []
    for artist in artist_rec_indices:
        print(data_artists.iloc[artist]['artists'])
        rec_artists.append(data_artists.iloc[artist]['artists'])

    input('Continue')

    for artist in rec_artists:
        print(pdata[pdata['artists'] == artist]['name'].head())

    input('Continue')


if __name__ == '__main__':

    # Authorization
    sp = auth.authorize_client()

    # Data reading
    pdata = pd.read_csv('Data/proc_data.csv')
    num_data = pd.read_csv('Data/num_data.csv')
    scaled_num_data = pd.read_csv('Data/scaled_num_data_df.csv')
    data_artists = pd.read_csv("Data/data_by_artist.csv")
    pdata.set_index(pdata['index'], inplace=True)
    scaled_num_data.set_index(scaled_num_data['index'], inplace=True)
    scaled_num_data.drop(['index'], axis=1, inplace=True)

    # Accepting song details for prediction
    search_name = input('Enter song name')
    search_artist = input('Enter artist(optional)')
    search_index, search_features = get_search(sp, search_name, search_artist)
    print(search_index, search_features)
    input('Entering predict, continute')
    predict(search_index, search_features)
    artist_rec(search_features)
