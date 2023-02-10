## >> This is about the API


  GET https://api.spotify.com/v1/audio-features/{id}


curl -X "GET" "https://api.spotify.com/v1/audio-features/6AzQZJLLmEUFurwExaWNjd" -H "Accept: application/json" -H "Content-Type: application/json" -H "Authorization: Bearer BQC1uCr_7QB-fx1uHpIWvnjzAckP9nd_TUhxxuFKlw43MAs3uOHNs3W4ZtqQK9pK1h_AuDWOcrCbbuLOc0I503ABZ10Qx-dXAtfB9mSFYgpDH6Gond_PEzZChi9LJGPjs8JAPliaNcW7arq6qXHsmjGZK-tjQEjuoOsbp3SEd414_BrWiq8uP9VKdZ87BGg6JLP028d0zuVmMU3JAJoHBj7FYJ6N7rm7LoxjzZq4pdQ"


BQC1uCr_7QB-fx1uHpIWvnjzAckP9nd_TUhxxuFKlw43MAs3uOHNs3W4ZtqQK9pK1h_AuDWOcrCbbuLOc0I503ABZ10Qx-dXAtfB9mSFYgpDH6Gond_PEzZChi9LJGPjs8JAPliaNcW7arq6qXHsmjGZK-tjQEjuoOsbp3SEd414_BrWiq8uP9VKdZ87BGg6JLP028d0zuVmMU3JAJoHBj7FYJ6N7rm7LoxjzZq4pdQ


https://developer.spotify.com/documentation/web-api/libraries/
https://spotipy.readthedocs.io/en/2.12.0/#authorization-code-flow

libraries
https://pypi.org/project/tekore/
https://github.com/plamere/spotipy



https://developer.spotify.com/console/get-audio-features-track/?id=6AzQZJLLmEUFurwExaWNjd




#####
# Dumped on the 28.04.2022 here: /Volumes/GoogleDrive/My Drive/SpotifyData

Request profile data (account/privacy settings)

Extracting Relevant Data
Once you have extracted the zip folder provided by Spotify, you will have access to information such as streaming history, personal details, and artists following, etc. The list of JSON files in the zip folder are as follows:
Follow.json — contains the following list as well as the current followers.
Identity.json — contains your information that is shown on the Spotify app such as name, photo, and verification etc.
Inferences.json — contains Spotify’s understanding of you as a user i.e., what kind of content you consume on Spotify like education, business, dance, and so on.
Payments.json — contains your payment information.
Playlist1.json — This JSON has the playlists information that is created by you.
SearchQueries.json — It stores your search history for example: at what time and on which system you have queried for a song, artist or podcast.
StreamingHistory0.json — This has your streaming history i.e., when and which song you have heard and for how long.
UserData.json — This contains the personal information that you provide at the time of sign up, for instance, your username, date of birth, email, gender, etc.
YourLibrary.json —The content that you have saved or liked can be found in this JSON file.
However, the JSON does not contain information like the date-music timestamp, the song id, song features, and other such info. So, to capture this information you can use a code written by Vlad Gheorghe. You can check out his data scraping code in his blog and Github.


# >> This is about getting info about myself following
# pip3 install git+https://github.com/mkurovski/liked2play/
# 
from liked2play.analysis import analyze_gdpr_data, read_liked_songs
from liked2play.utils import get_access_token
import logging
import time
import requests
import json
import pandas as pd
_logger = logging.getLogger(__name__)


dataFolder = '/Volumes/GoogleDrive-101555491803780988335/My Drive/SpotifyData'
dataFolder = '/Volumes/GoogleDrive/My Drive/SpotifyData'
config = {
  'gdpr_data_path': dataFolder,
  'interim_storage_folder': '/Users/carlos.aguilar/Documents/tempRubbish/spotify',
  'play_threshold': 60000,
}
analyze_gdpr_data(config, topn=30)


liked_songs = read_liked_songs(config['gdpr_data_path'])

# Check on Liking Behavior
liked_songs["id"] = liked_songs["uri"].apply(lambda val: val.split(":")[-1])
liked_songs["artist_track"] = [
    a + ": " + b
    for a, b in zip(liked_songs["artist"].values, liked_songs["track"].values)
]

track_ids = liked_songs["id"].values.tolist()
# audio_features = fetch_audio_features(track_ids, cfg)

client_id = 'eb5b3df8e05b432bb98923e710fb6e55'
client_secret = '0dfb4b2414424fe69a73871d7a8510f9'
access_token = get_access_token(client_id, client_secret)

headers = {"Authorization": "Bearer " + access_token}

base_url = "https://api.spotify.com"
endpoint = "/v1/audio-features"
url_feats = f"{base_url}{endpoint}"

batch_size = 100
sleep_seconds_between_batches: int = 5


results = list()
for idx in range(0, len(track_ids), batch_size):
    batch_ids = track_ids[idx : idx + batch_size]
    params = {"ids": ",".join(batch_ids)}
    _logger.info(
        f"Requesting Audio Features for Batch "
        f"[{idx}:{idx + batch_size}] / {len(track_ids)} ..."
    )
    res = requests.get(url=url_feats, headers=headers, params=params)
    res = json.loads(res.content)["audio_features"]
    assert len(batch_ids) == len(res)
    results.extend(res)

    time.sleep(sleep_seconds_between_batches)

audio_features = pd.DataFrame(results)
audio_features.index = track_ids

df_liked_songs = pd.merge(liked_songs, audio_features, left_on='id', right_index=True)

import carlos_utils.file_utils as fu
fu.toPickleFile(df_liked_songs, fu.fullfile(dataFolder, 'liked_songs_features.pickle'))

df_liked_songs.loc[2]


base_url = "https://api.spotify.com"
endpoint = "/v1/recommendations"
url_reco = f"{base_url}{endpoint}"
# params = {"seed_artists": ",".join(df_liked_songs.loc[2].id)}
params = {"seed_tracks": df_liked_songs.loc[2].id, 
'limit': 2, 'max_danceability': df_liked_songs.loc[2].danceability}
res = requests.get(url=url_reco, headers=headers, params=params)
res = json.loads(res.content)['tracks']
fu.printJSON(res)



##########
# directly from Spotify
# In Spotify, click on the song, share/copy link
# https://open.spotify.com/track/2mc92whzMSiIJQzi1MALDA?si=509fc50e28754d47
track_url = 'https://open.spotify.com/track/1t8TCORVxdItzE3zy1X0tv?si=023f61f7828f4a33'
track_url ='https://open.spotify.com/track/39LLxExYz6ewLAcYrzQQyP?si=b72b6ff61ebb4f4a'
track_id = track_url.split('?si=')[0].split('/')[-1]

base_url = "https://api.spotify.com"
endpoint = "/v1/audio-features"
url_feats = f"{base_url}{endpoint}"
endpoint = "/v1/recommendations"
url_reco = f"{base_url}{endpoint}"

endpoint = "/v1/me/player/queue"
url_playback_queue = f"{base_url}{endpoint}"

url_devices = f'{base_url}/v1/me/player/devices'

client_id = 'eb5b3df8e05b432bb98923e710fb6e55'
client_secret = '0dfb4b2414424fe69a73871d7a8510f9'
access_token = get_access_token(client_id, client_secret)
headers = {"Authorization": "Bearer " + access_token}


res = requests.get(url=url_devices, headers=headers)
res.ok



params = {"ids":track_id}
res = requests.get(url=url_feats, headers=headers, params=params)
track_feats = json.loads(res.content)['audio_features']
fu.printJSON(track_feats)

get_tolerance = lambda x, tol: [x*(1-tol), x*(1+tol)]

min_danceability, max_danceability = get_tolerance(track_feats[0]['danceability'], 0.1)

# get a similar song
params = {"seed_tracks": track_id, 
'limit': 12, 'min_danceability': min_danceability, 'max_danceability': max_danceability, }
res = requests.get(url=url_reco, headers=headers, params=params)

if res.ok:
  all_recos = []
  for this_recommendation in json.loads(res.content)['tracks']:
    album = this_recommendation['album']
    artist = album['artists'][0]['name']
    album_name = album['name']
    recommended_song_uri = this_recommendation['external_urls']['spotify']
    recommended_track_id = this_recommendation['id']
    recommended_track_name = this_recommendation['name']
    recommended_track_popularity = this_recommendation['popularity']

    d_reco = {'artist': artist,
      'album_name':album_name,
      'recommended_track_name': recommended_track_name,
      'recommended_track_id': recommended_track_id,
      'recommended_track_popularity': recommended_track_popularity,
      'recommended_song_uri': recommended_song_uri,
    }
    all_recos.append(d_reco)
    fu.printJSON(d_reco)


df_recommendations = pd.DataFrame(all_recos)


params = {"ids":track_id}
res = requests.post(url=url_playback_queue, headers=headers, params=params)

df_recommendations.iloc[-4]
# spotify:track:0SiywuOBRcynK0uKGWdCnn


## there are 3 ways of authorising
# https://developer.spotify.com/documentation/general/guides/authorization/

# pip3 install spotipy

url_login = 'https://accounts.spotify.com/authorize'
redirect_uri = 'http://localhost:8888/callback'
response_type = 'code'
client_id = 'eb5b3df8e05b432bb98923e710fb6e55'
client_secret = '0dfb4b2414424fe69a73871d7a8510f9'

scope = ['user-read-playback-state', 'playlist-modify-private', 
'playlist-modify-public', 'user-read-playback-position']

from spotipy.oauth2 import SpotifyOAuth
authenticator = SpotifyOAuth(client_id=client_id, 
client_secret=client_secret, redirect_uri=redirect_uri,
show_dialog=False,scope=scope)


auth_token_info = authenticator.get_access_token()
fu.printJSON(auth_token_info)
auth_token = auth_token_info['access_token']
len(auth_token)
pp.copy(auth_token)

authenticator.validate_token(auth_token_info)

auth_response = authenticator.get_auth_response()
pp.copy(auth_response)


authorization_code = authenticator.get_authorization_code()
len(authorization_code)
pp.copy(authorization_code)

a = 'BQCEgOI-SRKrHDpjZKKzzoYIkcbj-IY7pfdIxBJhy3xKcBn8bBFnf-epsLBW35hR1G4a6MuPlBdxW87VupQH34pKKIvekWe_7vNPYXB7IRBVr689bGnmDlLbsTepx4W6NPPgb63CEStJ22kohk-PbeqPu6xd6DGmP7VexIMoXMPPIwrXBmyX4xNUxymQ8vmxdgpYrWKedJbwAfXvb9P_tKnt7BDhk7AfS9DVzoUSNDxWLdy2qDL0rKdRcSoXwA3Q_v5DZoIrdQORVzHGxjGX'
len(a)
 




params = {"response_type": response_type, 
'client_id': client_id, 'redirect_uri': redirect_uri}
res = requests.get(url=url_login, params=params)
res.ok
res.text

client_secret = '0dfb4b2414424fe69a73871d7a8510f9'

access_token = 'BQAW5ZjPfYIH6mx6mm3uw-1idKZsggXtWy2YlnZDI9jaTBObiLMDhLd9Qwhh9bYKhWDsYUpHmedlagzRT4JZlrZPNVrYpPpkKWtAY8ZKkkG1PlgzhrRh1Rf_dkDs28clNBbahrFo4ehMvfdXrCjMJ_exmTeyUVjYcf6FmDyvctxav1s5UFLh-dsVVqJZF3Hex_9fIBq0zCi5mGOI_DsEgZqbVDgnDzw_61Uv7tvJcw7Umn5m8pydNsMN6FRVqibUZwYZ8HKiDhN7khgmZq9T'

headers = {"Authorization": "Bearer " + access_token}
