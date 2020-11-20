

# ----------------------DUMP-----------------------------

# OAuth = SpotifyOAuth(scope=scope,
#                      redirect_uri= redirect_uri,
#                      cache_path='../cache.txt')
# token = OAuth.get_access_token()

# results = sp.search(q='weezer', limit=20)
# for idx, track in enumerate(results['tracks']['items']):
#   print(idx, track['name'])
#
# name = 'Radiohead'
# results_radio = sp.search(q='artist:' + name, type='artist')
# items = results_radio['artists']['items']
# # results_radio
# if len(items) > 0:
#     artist = items[0]
#     print(artist['name'], artist['images'][0]['url'])
#     webbrowser.open(artist['images'][0]['url'], new= 2)

# # train_data = pd.read_csv("/content/drive/My Drive/SpotiRecSys")
# import os
#
# root = "/content/drive/My Drive/SpotiRecSys/160k"
# csvs = []
# dfs = []
#
# for dirpath, dirnames, filenames in os.walk(root):
#     for file in filenames:
#         csvs.append(dirpath + '/' + file)
#
# for f in csvs:
#     if f.endswith('.csv'):
#         dfs.append(pd.read_csv(f))
#
# dfs
#
# dfs[1].head(15)


# #Authorization from Spotify
# OauthParams = {'client_id' : client_id, 'response_type' : 'code', 'redirect_uri' : redirect_uri}
# get = requests.get("https://accounts.spotify.com/authorize",params = OauthParams)
# webbrowser.open(get.url,new=2)

# # http://127.0.0.1:9090/?code=AQC4Xa9v0qxOlauM1QBbs3VzdXEs8q71dhLuSRQfKl7082BEBMU8EqdM3U0aGpF3cFzN9VZT7eeosEm3rt55nQJmdUvGTvby1vuLrcNUx1C0W7-bWwl_F_E2iRk5renV2-hqAU81k4pMJ86dGSmasSi0sTJXlr--nA
# code = 'AQC4Xa9v0qxOlauM1QBbs3VzdXEs8q71dhLuSRQfKl7082BEBMU8EqdM3U0aGpF3cFzN9VZT7eeosEm3rt55nQJmdUvGTvby1vuLrcNUx1C0W7-bWwl_F_E2iRk5renV2-hqAU81k4pMJ86dGSmasSi0sTJXlr--nA'
#
# PostBodyData = {'grant_type' : "authorization_code", 'code' : code , 'redirect_uri' : redirect_uri, 'client_id' : client_id, 'client_secret' : client_secret}
# #PostData = {'access_token' : code , 'token_type' : 'bearer', }
# post = requests.post("https://accounts.spotify.com/api/token", data = PostBodyData)
# post.content
#
# string_post = post.content.decode('utf-8')
# access_json = json.loads(string_post)
# access_json
#
# access_token = access_json['access_token']
# access_token


# TEST READ LIBRARY
