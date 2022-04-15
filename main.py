# import modules, prep spotipy oauth
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import cred
import time
import random
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import distance as dist
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.compose import ColumnTransformer
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import Counter
from pyclustering.cluster.kmeans import kmeans as pykmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric


scope = "playlist-read-private, playlist-modify-public, user-read-private, user-top-read, user-library-read"
auth_manager = SpotifyOAuth(client_id=cred.client_id, client_secret=cred.client_secret, redirect_uri='http://127.0.0.1:8080', scope=scope)
sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10, retries=5)

user_id = sp.current_user()['id']
user_country = sp.current_user()['country']
username = sp.current_user()['display_name']
if sp.current_user()['explicit_content']['filter_enabled'] or sp.current_user()['explicit_content']['filter_locked']:
  user_explicit = False
else:
  user_explicit = True


# Functions

def get_user_top_tracks():

  long_term_tracks = sp.current_user_top_tracks(time_range='long_term', limit=30)
  medium_term_tracks = sp.current_user_top_tracks(time_range='medium_term', limit=30)
  short_term_tracks = sp.current_user_top_tracks(time_range='short_term', limit=30)

  top_track_ids = []
  for ltrack in long_term_tracks['items']:
    top_track_ids.append(ltrack['uri'])

  for mtrack in medium_term_tracks['items']:
    top_track_ids.append(mtrack['uri'])

  for strack in short_term_tracks['items']:
    top_track_ids.append(strack['uri'])

  return list(set(top_track_ids))

def get_user_playlist_ids():
  '''
  Collects a list of user playlist dictionaries and the Spotify ID for each of them.
  '''
  playlists_lst =[]
  ids = []
  offset = 0
  while True:
      playlists = sp.current_user_playlists(offset=offset)
      if len(playlists['items']) == 0:
          break
      for playlist in playlists['items']:
          playlists_lst.append(playlist)
      offset = offset + len(playlists['items'])
      time.sleep(0.0001) 
  
  for playlist in playlists_lst:
      ids.append(playlist['id'])
  return ids, playlists_lst

def get_saved_tracks():
  '''
  Gets a user's "liked" tracks
  '''
  ids = []
  print('I\'m starting to look at the user\'s saved tracks!!')
  offset = 0
  t1 = time.time()
  while True:
      track_ids = sp.current_user_saved_tracks(offset=offset)
      if len(track_ids['items']) == 0:
          break
      for track in track_ids['items']:
          if track['track'] == None:
              continue
          else:
              ids.append(track['track']['id'])
      offset = offset + len(track_ids['items'])
      time.sleep(0.0001)
  t2 = time.time()
  print(f'Hmmm... getting the liked tracks took {t2-t1} seconds!\n')
  return list(set(ids))

def get_playlist_names(playlists):
  '''
  Returns a list of a user's playlist titles when given a list of playlist ids
  '''
  names = []
  for playlist in playlists:
      name = sp.playlist(playlist)['name']
      names.append(name)
  return names

def get_song_ids_from_playlists(user, playlist_urls):
  '''
  Gets song ids from each of the songs in given playlist ids
  '''
  ids = []
  t1 = time.time()
  for i in range(len(playlist_urls)):
      offset = 0
      playlist_name = get_playlist_names([playlist_urls[i]])
      print(f'I\'m grabbing saved songs from playlist number {i+1} out of {len(playlist_urls)}: {playlist_name[0]}')
      while True:
          track_ids = sp.user_playlist_tracks(user=user, playlist_id=playlist_urls[i], offset=offset, fields ='items.track.id')
          #print(track_ids)
          #print(len(track_ids['items']))
          if len(track_ids['items']) == 0:
              break
          for track in track_ids['items']:
              if track['track'] == None:
                  continue
              else:
                  ids.append(track['track']['id'])
          offset = offset + len(track_ids['items'])
          time.sleep(0.0001)
  t2 = time.time()
  print(f'Getting song ids from all those playlists took {round(t2-t1, 2)} seconds!\n')
  return list(set(ids))

def get_recc_ids(list_seed_tracks, country):
  '''
  Gets ids for # recommended songs for each song in the seed tracks list
  '''
  print('Starting to collect recommendation ids.')
  if len(list_seed_tracks) > 150:
    print(f'Wow! I have {len(list_seed_tracks)*20} to make. This may take a while.\n')

  recc_ids = []
  #raw_recs = []
  t1 = time.time()
  for seed in list_seed_tracks:
    seed_to_use = []
    seed_to_use.append(seed)
    recs = sp.recommendations(seed_tracks=seed_to_use, limit = 30, country=country)
    #raw_recs.append(recs)
    #print(recs)
    for i in range(len(recs['tracks'])):
      track_id = recs['tracks'][i]['id']
      if track_id not in recc_ids:
        recc_ids.append(track_id)
    #print(len(recc_ids))
  set_ids = set(recc_ids) # extra check to make sure there are no duplicates
  t2 = time.time()
  print(f'Making and saving all of those recommendations took {round(t2-t1, 2)} seconds.\n')
  return list(set_ids)

def create_playlist(tracks):
  sp.user_playlist_create(user_id, 'your recommended songs', description='yay new songs!')
  user_playlists, y = get_user_playlist_ids()
  sp.user_playlist_add_tracks(user_id, user_playlists[0], tracks)
  return 'Your playlist has been created!'

def create_df(track_ids, in_lib):
  print(f'{len(track_ids)} observations to make!')
  data = []

  for i in range(len(track_ids)):
    # Get raw data for track
    try:
      track = sp.track(track_ids[i])
      features = sp.audio_features(track_ids[i])
      analysis = sp.audio_analysis(track_ids[i])
      artist_uri = track['album']['artists'][0]['uri']
      artist = sp.artist(artist_uri)
      decade_prep = track['album']['release_date'][0:3]
      decade = int(decade_prep + '0')
      if int(track['album']['release_date'][3]) >= 0 and int(track['album']['release_date'][3]) < 5:
        half_decade = int(track['album']['release_date'][0:3] + '0')
      else:
        half_decade = int(track['album']['release_date'][0:3] + '5')
    except:
      print(f'skipped one ({in_lib})!')
      continue

    
    
    # Extract relevant data
    observation = [
      track['uri'], 
      track['name'],
      in_lib,
      artist['followers']['total'],
      artist['genres'],
      artist['popularity'],
      track['explicit'],
      track['album']['release_date'][0:4], 
      int(track['album']['release_date'][0:4]),
      decade,
      half_decade,
      len(track['artists']),
      round(track['duration_ms']/60000, 4),
      track['popularity'],
      features[0]['danceability'],
      features[0]['energy'],
      features[0]['key'],
      analysis['track']['key_confidence'],
      features[0]['loudness'],
      features[0]['mode'],
      analysis['track']['mode_confidence'],
      features[0]['speechiness'],
      features[0]['acousticness'],
      features[0]['instrumentalness'],
      features[0]['liveness'],
      features[0]['valence'],
      features[0]['tempo'],
      analysis['track']['tempo_confidence'],
      features[0]['time_signature'],
      analysis['track']['time_signature_confidence'],
      analysis['track']['num_samples'],
      len(analysis['bars']),
      len(analysis['beats']),
      len(analysis['sections']),
      len(analysis['segments']), # for each segment, there is a list of pitches and timbre!
      len(analysis['tatums'])
    ]

    # Add observation to total dataset
    data.append(observation)
    time.sleep(0.00000001)
    
  # Create final data frame with proper column names
  df = pd.DataFrame(data, columns=[
    'uri', 'track_name', 'in_library', 'artist_followers', 'genre', 'artist_popularity', 'explicit', 'release_date', 
    'year', 'decade', 'half_decade', 'nartists', 'duration_m', 'track_popularity', 'danceability', 'energy',
    'key', 'key_conf', 'loudness', 'mode', 'mode_conf', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'tempo_conf', 'time_sig', 'time_sig_conf', 'nsamples', 'nbars',
    'nbeats', 'nsections', 'nsegments', 'ntatums'
  ])

  return df

def df_manage(reccs, saved):
  recc_df = create_df(reccs, 0) # df of recommendations
  saved_df = create_df(saved, 1) # df of songs in library
  combined_df = pd.concat([recc_df, saved_df], ignore_index = True) # the previous two combined
  return recc_df, saved_df, combined_df

def count_predict(kmeans, user_weights, group, clusters, recc_df, saved_df, num_feat, ord_feat, nom_feat):
  
  scaled_reccs, x = scaling(group, recc_df, user_weights, num_feat, ord_feat, nom_feat)
  scaled_saved, x = scaling(group, saved_df, user_weights, num_feat, ord_feat, nom_feat)

  # Predict clusters for recommendations
  recc_predictions = kmeans.predict(scaled_reccs)
  
  # Predict clusters for saved tracks
  saved_predictions = kmeans.predict(scaled_saved)
  # print(saved_predictions)

  # Set counts for all recc clusters to 0 (ensures that all clusters are present)
  initial_recc_counts = {}
  for i in range(0, clusters):  
    initial_recc_counts[i] = 0

  # Create a counter and add in predicted cluster counts for recommendations
  cluster_recc_counts = Counter(initial_recc_counts)
  cluster_recc_counts.update(recc_predictions)

  # Set counts for all saved clusters to 0 (ensures that all clusters are present)
  initial_saved_counts = {}
  for i in range(0, clusters):  
    initial_saved_counts[i] = 0

  # Create a counter and add in predicted cluster counts for saved tracks
  cluster_saved_counts = Counter(initial_saved_counts)
  cluster_saved_counts.update(saved_predictions)

  # Create new dict with percentages of songs in library that are in each of the clusters
  cluster_prop = {}
  for item in cluster_saved_counts:
    cluster_prop[item] = cluster_saved_counts[item] / saved_df.shape[0]

  n_to_recc = {}
  for item in cluster_prop:
    n_to_recc[item] = round(cluster_prop[item]*30, 7)
  
  return recc_predictions, cluster_recc_counts, cluster_saved_counts, cluster_prop, n_to_recc

def add_cluster(recc_data, saved_data, clusters):
  recc_with_clusters = recc_data.assign(cluster = clusters)
  saved_with_clusters = saved_data.assign(cluster = max(clusters) + 1)
  new_combined = pd.concat([recc_with_clusters, saved_with_clusters], ignore_index = True)
  return recc_with_clusters, new_combined

def silhouette_plot(algo, dist_metric, scaled_og_data):
  range_n_clusters = [2, 3, 4, 5, 6]

  for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(scaled_og_data) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    initial_centers = kmeans_plusplus_initializer(data = scaled_og_data, amount_centers = n_clusters).initialize()
    clusterer = pykmeans(scaled_og_data, initial_centers, metric = dist_metric)
    clusterer.process()
    #clusters = kmeans_instance.get_clusters()
    #final_centers = kmeans_instance.get_centers()
    # type(final_centers[0])
    cluster_labels = clusterer.predict(scaled_og_data)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(scaled_og_data, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(scaled_og_data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
      # Aggregate the silhouette scores for samples belonging to
      # cluster i, and sort them
      ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

      ith_cluster_silhouette_values.sort()

      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      color = cm.nipy_spectral(float(i) / n_clusters)
      ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
      )

      # Label the silhouette plots with their cluster numbers at the middle
      ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      # Compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
      scaled_og_data[:, 0], scaled_og_data[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    if algo == 'sklearn':
      centers = clusterer.cluster_centers_
    elif algo == 'pycluster':
      centers = np.array(clusterer.get_centers())
    # Draw white circles at cluster centers
    ax2.scatter(
      centers[:, 0],
      centers[:, 1],
      marker="o",
      c="white",
      alpha=1,
      s=200,
      edgecolor="k",
    )

    for i, c in enumerate(centers):
      ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 3rd feature")
    ax2.set_ylabel("Feature space for the 4th feature")

    plt.suptitle(
      "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
      % n_clusters,
      fontsize=14,
      fontweight="bold",
    )

  plt.show()

def diagnostics(algo, metric, scaled_data, fit_data, vars, pairplot, silhouette, elbow):
  
  if pairplot:
    if len(vars) == 2:
      sns.scatterplot(data = fit_data, x = vars[0], y = vars[1])
      plt.show()

      sns.scatterplot(data = fit_data, x = vars[0], y = vars[1], hue = 'cluster', palette = 'Spectral')
      plt.show()
    else:
      sns.pairplot(data = fit_data, vars = vars, hue = 'cluster', palette = 'Spectral')
      plt.show()
  
  if silhouette:
    silhouette_plot(algo, metric, scaled_data)

  # if elbow and algo == 'sklearn':
  #   distortions = []
  #   for k in range(2, 15):
  #     kmeans = KMeans(n_clusters=k, random_state=10, init='k-means++')
  #     kmeans.fit(scaled_data)
  #     distortions.append(kmeans.inertia_)

  #   fig = plt.figure(figsize=(15, 5))
  #   plt.plot(range(2, 15), distortions)
  #   plt.grid(True)
  #   plt.title('Elbow curve')
  
def detect_outlier(scaled_data, combined_df):
  from sklearn.ensemble import IsolationForest
  iso = IsolationForest(contamination=0.1)
  yhat_iso = iso.fit_predict(scaled_data)
  obs_iso = np.where(yhat_iso == -1)[0]

  from sklearn.neighbors import LocalOutlierFactor
  lof = LocalOutlierFactor()
  yhat_lof = lof.fit_predict(scaled_data)
  obs_lof = np.where(yhat_lof == -1)[0]

  from sklearn.svm import OneClassSVM
  ee = OneClassSVM(nu=0.01)
  yhat_ee = ee.fit_predict(scaled_data)
  obs_ocs = np.where(yhat_ee == -1)[0]

  common_outliers = sorted(set(obs_iso).union(set(obs_lof), set(obs_ocs)))
  # print(common_outliers)
  recc_outliers = []
  for index in common_outliers:
    if combined_df.iloc[index]['in_library'] == 0:
      recc_outliers.append(index)

  print(f'Nice! There are {len(recc_outliers)} outliers.')
  return recc_outliers

def manhattan(object1, object2):
  return dist.cityblock(object1, object2)

def chebyshev(object1, object2):
  return dist.chebyshev(object1, object2)

def minkowski6(object1, object2):
  if len(object1.shape) > 1 or len(object2.shape) > 1:
    return np.power(np.sum(np.power(object1 - object2, 6), axis=1), 1/6)
  else:
    return np.power(np.sum(np.power(object1 - object2, 6)), 1 / 6)

def minkowski12(object1, object2):
  if len(object1.shape) > 1 or len(object2.shape) > 1:
    return np.power(np.sum(np.power(object1 - object2, 12), axis=1), 1/12)
  else:
    return np.power(np.sum(np.power(object1 - object2, 12)), 1 / 12)

def canberra(object1, object2):
  return dist.canberra(object1, object2)

def euclidean(object1, object2):
  return dist.euclidean(object1, object2)

# Get all playlist ids and all complete playlist information
playlist_ids, raw_playlists = get_user_playlist_ids()
print(f'Cool! You have {len(playlist_ids)} playlists in your library!')

playlist_names = get_playlist_names(playlist_ids)

def show_playlists(playlist_ids):

  numbered_names = []
  for i in range(len(playlist_ids)):
    numbered_name = str(i + 1) + '.' + ' ' + playlist_names[i]
    numbered_names.append(numbered_name)
  for name in numbered_names:
    print(name)

while True:

  print(f'I see {len(playlist_ids)} playlists in your library! But you may not want all of those to be used for recommendations. Type in the number of the playlist you want removed. When all you\'re finished, type \"Done\".')

  show_playlists(playlist_ids)

  list_to_remove = input('Type number here!')
  print(list_to_remove)

  if list_to_remove == 'Done':
    break
  else:
    del playlist_ids[int(list_to_remove) - 1]
    del playlist_names[int(list_to_remove) - 1]

# Get the song ids for all tracks in the selected playlist(s)
list_ids = get_song_ids_from_playlists(user_id, playlist_ids)
liked_ids = get_saved_tracks()
saved_ids = list_ids + liked_ids
print(f'Whoa...you have {len(saved_ids)} songs in your library (using the playlists provided).')

recc_ids = get_recc_ids(saved_ids, user_country)

def remove_explicit(recc_ids):
  for recc in recc_ids:
    if sp.track(recc)['explicit']:
      recc_ids.remove(recc)

if not user_explicit:
  remove_explicit(recc_ids)

top_track_ids = get_user_top_tracks()

for top_track in top_track_ids:
  if top_track not in saved_ids:
    saved_ids.append(top_track)
    saved_ids.append(top_track)
  else:
    continue

# Build data frames
# ASSUMING 30 RECOMMENDATIONS PER SONG IN LIBRARY

recc_df, saved_df, combined_df = df_manage(recc_ids, saved_ids)

# Scale all data, fit new kmeans

# Group 1: Random 30 basic reccs

# Group 2: Euclidean distance with only numeric vars
# Group 3: Euclidean distance with all vars
# Group 4: Random other distance with only numeric vars
# Group 5: Random other distance with all vars

# Group 6: Euclidean distance with only numeric vars and user weights
# Group 7: Euclidean distance with all vars and user weights
# Group 8: Random other distance with only numeric vars and user weights
# Group 9: Random other distance with all vars and user weights

groups = [1, 2, 3, 4, 5, 6, 7, 8, 9]
distances = [manhattan, canberra, chebyshev, minkowski6, minkowski12]

user_group = random.choice(groups)
if user_group in [4, 5, 8, 9]:
  user_distance = random.choice(distances)
else:
  user_distance = euclidean

if user_group in [6, 7, 8, 9]:
  danceability_weight = input('On a scale of 1 to 5, how important is it to you that a song be \"danceable\"? (1 = not important, 5 = very important)')
  valence_weight = input('On a scale of 1 to 5, how important is it to you that a song be happy/upbeat? (1 = not important, 5 = very important)')
  acousticness_weight = input('On a scale of 1 to 5, how important is it to you that a song be acoustic? (1 = not important, 5 = very important)')
  liveness_weight = input('On a scale of 1 to 5, how important is it to you that a song be live? (1 = not important, 5 = very important)')
  energy_weight = input('On a scale of 1 to 5, how important is it to you that a song be energetic? (1 = not important, 5 = very important)')
  user_weights = [1, 1, 1, int(danceability_weight), int(energy_weight), 1, int(acousticness_weight), 1, int(liveness_weight), 1, int(valence_weight)]

def scaling(group, combined_df, weights, numeric_features, ordinal_features, nominal_features):

  num_transformer = StandardScaler()
  ordnom_transformer = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(), ordinal_features),
            ('nom', OneHotEncoder(), nominal_features)
        ]
    )
  
  if group in [2, 4]:
    scaled_data = num_transformer.fit_transform(combined_df[numeric_features])
    all_vars = numeric_features
  elif group in [6, 8]:
    scaled_data = num_transformer.fit_transform(combined_df[numeric_features])
    for i in range(len(weights)):
      scaled_data[:, i] *= weights[i]**(1/2)
    all_vars = numeric_features
  
  elif group in [3, 5]:
    scaled_num = num_transformer.fit_transform(combined_df[numeric_features])
    scaled_ordnom = ordnom_transformer.fit_transform(combined_df[ordinal_features + nominal_features])
    scaled_data = np.concatenate((scaled_num, scaled_ordnom), axis = 1)
    all_vars = numeric_features + ordinal_features + nominal_features
  elif group in [7, 9]:
    scaled_num = num_transformer.fit_transform(combined_df[numeric_features])
    scaled_ordnom = ordnom_transformer.fit_transform(combined_df[ordinal_features + nominal_features])
    for i in range(len(weights)):
      scaled_num[:, i] *= weights[i]**(1/2)
    scaled_data = np.concatenate((scaled_num, scaled_ordnom), axis = 1)
    all_vars = numeric_features + ordinal_features + nominal_features
  
  return scaled_data, all_vars

def kmeans_process(distance, group, user_weights, combined_df, recc_df, saved_df):

  if distance in ['chebyshev', 'minkowski_6', 'euclidean', 'minkowski_12']:
    clusters = 3
  else:
    clusters = 2

  if group in [2, 4, 6, 8]:
    numeric = True
  else:
    numeric = False

  numeric_features = ['artist_popularity',
    'duration_m', 'track_popularity', 'danceability', 'energy',
    'loudness', 'acousticness', 'instrumentalness',
    'liveness', 'tempo', 'valence']
  ordinal_features = ['half_decade', 'key', 'time_sig']
  if user_explicit:
    nominal_features = ['explicit', 'mode']
  else:
    nominal_features = ['mode']

  scaled_data, all_vars = scaling(group, combined_df, user_weights, numeric_features, ordinal_features, nominal_features)
  
  metric = distance_metric(type_metric.USER_DEFINED, func=distance)

  initial_centers = kmeans_plusplus_initializer(data = scaled_data, amount_centers = clusters).initialize()
  kmeans = pykmeans(scaled_data, initial_centers, metric = metric)
  kmeans.process()

  # recc_clusters, recc_count, saved_count, saved_prop, nto_recc = count_predict(kmeans, numeric, recc_df, saved_df, all_vars, numeric_features, ordinal_features, nominal_features)
  # recc_with_cluster, combined_with_cluster = add_cluster(recc_df, saved_df, recc_clusters)

  new_recc_df = recc_df.drop(detect_outlier(scaled_data, combined_df), axis = 0, inplace = False)
  new_combined_df = pd.concat([new_recc_df, saved_df], ignore_index = True)
  new_scaled_data, _ = scaling(group, new_combined_df, user_weights, numeric_features, ordinal_features, nominal_features)

  new_initial_centers = kmeans_plusplus_initializer(data = new_scaled_data, amount_centers = clusters).initialize()
  new_kmeans = pykmeans(new_scaled_data, new_initial_centers, metric = metric)
  new_kmeans.process()

  new_recc_clusters, new_recc_count, new_saved_count, new_saved_prop, new_nto_recc = count_predict(new_kmeans, user_weights, group, clusters, new_recc_df, saved_df, numeric_features, ordinal_features, nominal_features)

  new_recc_df_clusters, new_ncombined = add_cluster(new_recc_df, saved_df, new_recc_clusters)

  return new_scaled_data, new_recc_clusters, new_nto_recc, new_ncombined, all_vars, metric

new_scaled, new_recc_clusters, new_nto_recc, new_ncombined, all_vars, metric = kmeans_process(user_distance, user_group, user_weights, combined_df, recc_df, saved_df)

# diagnostics('pycluster', metric, new_scaled_data, new_recc_clusters, all_vars, False, True, True)

# Recommendations

def group1(og_reccs, in_lib_tracks):

  random.shuffle(og_reccs)

  if len(og_reccs) <= 30:
    group1 = og_reccs
  else:
    group1 = []
    for i in range(len(og_reccs)):
      if og_reccs[i] not in in_lib_tracks:
        group1.append(og_reccs[i])
      if len(group1) == 30:
        break

  return group1

def other_groups(to_recc_counts, clustered_data, in_lib_tracks):
  group2 = []
  
  for cluster in to_recc_counts.keys():
    one_cluster = clustered_data[clustered_data.cluster.eq(cluster)] # Select one cluster
    uris = list(one_cluster['uri']) # Select only uris
    reccs = random.sample(uris, k = round(to_recc_counts[cluster])) # take a random sample of the correct size
    for i in range(len(reccs)):
      group2.append(reccs[i]) # Append uris to group2

  # Remove any tracks already in library
  for i in range(len(group2)):
    if group2[i] in in_lib_tracks:
      group1.remove(group2[i])
  return group2

if user_group == 1:
  create_playlist(group1(recc_ids, saved_ids))
else:
  create_playlist(other_groups(new_nto_recc, new_ncombined, saved_ids))