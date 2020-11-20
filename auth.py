import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import os

client_id = "7d7875d6e08249ed8a047da889fac81f"
client_secret = "2f7cd0462cdb45a881a8bea4fd0504d8"
redirect_uri = "http://127.0.0.1:9090"

def authorize_client():
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def authorize_user():
    scope = "user-library-read user-top-read playlist-read-private playlist-read-collaborative"
    sp = spotipy.Spotify(client_credentials_manager=SpotifyOAuth(scope=scope, show_dialog=True))
    return sp

