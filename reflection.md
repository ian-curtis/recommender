# A Recommendation Algorithm for Spotify Users

Spotify users may be familiar with current Spotify-provided personalized recommendation systems. One, Discover Weekly provides listeners with new songs they haven't heard before and another provides recently-released songs that a listeners might enjoy. Both playlists give 30 songs one time a week and are the result of a complex algorithm. The exact methods used to construct this algorithm are private to Spotify as are the variables considered in recommendation. Furthermore, Spotify has access to a substantially larger amount of personalized data than developers such as myself to use to determine songs to recommend.

Why then does this project exist? Why build a recommendation algorithm for Spotify why 

Music is one of the rare forms of communication that can be understood on a profound level by anyone; it has the power to cause significant emotional effects, to spark inspiration, to ignite change, to spread knowledge, and more, even regardless of song language. Music’s influence is ever-growing, and its unique powers have become significantly more relevant throughout the COVID-19 pandemic. As communities shut down and governments issued lockdown orders, music played a more significant role for many people wishing to take advantage of their isolation and extra time. Studies in Spain and Israel have demonstrated that, during the pandemic, people have turned to music more than average, whether it be listening to songs, playing an instrument, or watching a livestream (Cabedo, Ziv). 


The goal of this program are to provide a new way for users to find new music without having to wait a whole week and to make those recommendations as accurate as possible. In other words, we want to minimize the number of recommended songs that a user did not enjoy and to recommend as many songs as possible that a user will enjoy. A few sub-goals follow: 

* We want to avoid recommended only Top-40 songs (unless this is only what a user has saved)
* We want to recommend songs that a user does not already have saved in their library
* We want to recommend songs that a user has not heard before (see Setbacks below for more on this)

As detailed in Future Work below, this program could also serve as a centerpiece to numerous research projects dealing with customizing features on an individual basis or with understanding how different forms of recommendation can affect how likely a user is to enjoy recommended products.

## Prerequisites

This program is not currently available on the web and is not able to be run by an individual user. The app requires an app ID and an app "password", both of which are keep secret for security purposes. See Future Work below for more information.

Should anyone wish to run the app for themselves, they would need to have an installation of Python, a nice text editor (such as Visual Studio Code), and a familiarity with how to use Git/GitHub and run a Python program. They then might create clone this repository with 

```
git clone https://github.com/ian-curtis/recommender.git
```

then make a developer account on Spotify, create a new file `cred.py` with thie following:

```
client_id = <your-app-id>
client_secret = <your-app-secret>
```

where the values in brackets would be replaced with a string of your newly-created app's ID and secret. The app should run as expected, provided that the user has a Spotify account and preferably at least 30 songs saved in their library (i.e., in playlists), although the app will still function with as low as 1 song saved (in a playlist).


## The App: How It Works

The program in `main.py` contains code for an algorithm that recommends new songs to Spotify users. Identical code is found in `main.ipynb` but is broken up into mangageable chunks so that the entire program does not have to be run at once.

What happens when the program is run depends on random choice. The program is designed to give users a different experience by changing the method by which recommendations are created so when the program is run, the user is assigned to a random group. Regardless of group assignment, all users will go through the following first steps:

1. The program loads all necessary outside packages, connects to the Spotify API (online database from which user data can be accessed), and sets up user information (username, country, and whether they can listen to explicit music).

2. The program reads in all of a user's playlists then asks the user if there are any playlists they would *not* like considered in the recommendation process. For instance, a user may want their holiday playlists excluded from the algorithm.

3. For each playlist to be analyzed, the program collects all of the songs in the playlist and removes the duplicates, creating a master list of all songs a user is known to "like" (or like enough to add to their library).

4. For each song in the master list, 30 recommendations are generated from a built-in recommendation algorithm and duplicates are removed. (Note: the algorithm used to generate these recommendations is not known.)

It is not necessary to recommend all of the songs in this list (e.g., for the author, this would mean recommending at least 20,000 songs). Thus the goal is to now determine the best songs out of the list to recommend. 

It is at this point that groups may differ. What follows are the specific steps taken by the program to create recommendations for each respective random group.

### Group 1: Random Reccomendations

This group is the simplest and is the least effective at providing songs a user will actually enjoy. Continuing from Step 4 above,

5. Out of the list of generated recommendations, 30 are randomly selected and given to the user.

### Group 2: KMeans With Euclidean Distance

This group uses the KMeans algorithm on certain attributes from songs saved in a user's library. KMeans is a clustering algorithm in that it uses given data and attempts to find natural clusters in the data. Thus, when presented with an observation it hasn't seen before, it will use the clusters it created to predict where the new observation lies. 

The creation of the clusters revolves around the distance between points. MORE HERE

Continuing from Step 4 above,

5. For each song saved in the user's library *and* for each potential song to be recommended, Spotify-provided attributes are collected. These include "danceability", "energy", "tempo", "key", and more. These are temporarily stored in a dataframe.

6. Variables are standardized so that no variable has more weight than another.

7. A KMeans algorithm is fit to the standardized data.

## Rationale

## Data and Permissions Required

In order to run this app, a user must give it permission to access their profile data as well as the songs in their library. The app can only access the information a user permits it too. As such, I have chosen the permissions necessary to make the app run ensuring that no excess data is used. This app uses the following scopes with the reasoning behind each scope next to it:

| Scope Name               | Reason                                                                                                                                            |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `user-library-read`      | Allows the app to collect songs in a user's library from playlists marked as "public" or "collaborative"                                          |
| `playlist-read-private`  | Allows the app to collect songs in a user's library from playlists marked as "private"                                                            |
| `playlist-modify-public` | Allows the app to create a new playlist, modify the title/description, and add new songs                                                          |
| `user-read-private`      | Allows the app to read private profile data. The app only uses information user country and whether or a not a user can listen to explicit music. |
| `user-top-read`          | Allows the app to read a user's top tracks and top artists                                                                                        |

At the end of the program, all user data and track information is erased. Should a user want more recommendations, they would have to run the app again. The app would not modify the old recommendation playlist but will instead create a new one.


## Setbacks

* The proportion of songs in their library might not represent the actual proporation of songs they listen to or like
* Don't have access to Spotify data (like language or songs a user listens to)
* Don't know how recommendations were generated
* Don't have access to all songs on Spotify to just select from
* Data is from Spotify and are likely to be accurate but may not be in some cases

## Future Work / Potential Applications

Although this program