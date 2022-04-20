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

## Future Work
* Incorporate the code into a web application where users can run the app without installing Python (this may involve switching to R and Shiny apps)
* Build in a way to include track genres
* Support data for tracks with more than one artist
* Add in small facts about a user's library and top tracks while they wait for the algorithm to finish (in progress)
* Add more possible distance metrics
* Reduce the time it takes to complete the recommendation process
* Adding more specific comments throughout the code (especially functions) to help guide other developers
* Add error handling mechanisms to avoid unnessecary stopping
* Build in a semi-supervised learning option incorporating user feedback to generate a new playlist
* Improve the code that removes any to-be-recommended songs already in a user's library
  * Currently relies on unique track ids but it is possible for the same track to have a different id, such as Blank Space from Taylor Swift's 1989 and Blank Space from Taylor Swift's 1989 (Deluxe)
* Turn code into a more object-oriented approach