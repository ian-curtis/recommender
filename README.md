# Recommendations for Spotify Users

The program found in `main.py` contains code for an algorithm that recommends new songs to Spotify users. Identical code is found in `main.ipynb` but is broken up into mangageable chunks so that the entire program does not have to be run at once. The process is explained in more detail below in the `reflection.md` file.

To run the program, clone this repository with

```
git clone https://github.com/ian-curtis/recommender.git
```

then make a (https://developer.spotify.com/)[developer] account on Spotify, create a new file `cred.py` with thie following:

```
client_id = <your-app-id>
client_secret = <your-app-secret>
```

where the values in brackets would be replaced with a string of your newly-created app's ID and secret. The app should run as expected, provided that the user has a Spotify account and preferably at least 30 songs saved in their library (i.e., in playlists), although the app will still function with as low as 1 song saved.