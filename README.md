# Spotify User Satisfaction With Recommended Playlists

This project was my submission to the Grand Valley State University Honors College senior project under mentor Dr. Erin Carrier. All code in this repository is licensed under an [MIT license](LICENSE).

The program found in `main.py` contains code for an algorithm that recommends new songs to Spotify users. Identical code is found in `main.ipynb` but is broken up into mangageable chunks so that the entire program does not have to be run at once. The process is explained in more detail below in the [reflection](reflection.md) file. Note that there may be differences between `testing.ipynb` and `main.py`/`main.ipynb`. The most up-to-date and working code is in the `main.*` files; `testing.ipynb` is used for testing new features and the effectiveness of the algorithm. A planned feature is to have the algorithm give users random fun facts about their listening habits. This is not currently implemented, but when it is, necessary code will update in `facts.ipynb` and `facts.py`. (Note that this will likely require a conversion to object-oriented programming, another planned feature.)

To run the program, clone this repository with

```
git clone https://github.com/ian-curtis/recommender.git
```

then make a [developer account on Spotify](https://developer.spotify.com/), create a new file here called `cred.py` with the following:

```
client_id = <your-app-id>
client_secret = <your-app-secret>
```

where the values in brackets would be replaced with a string of your newly-created app's ID and secret. The app should run as expected, provided that the user has a Spotify account and preferably at least 30 songs saved in their library (i.e., in playlists), although the app should still function with as low as 1 song saved.
