"""
K-Means clustering for a subset of all historic and current NBA players.
Clustering uses career average PER GAME stats.
Finds the optimal number of clusters by maximizing the silhouette score.

NOTE: the final number of clustered players can be less than the sample size
because some players returned have no stats for which to cluster.

Usage: python3 cluster.py

By: Jack Swisher
"""

import pandas as pd
import numpy as np
import requests
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MaxAbsScaler

SAMPLE_SIZE = 50
SAMPLE_DOWNLOAD_LINK = 'https://stats.nba.com/stats/commonallplayers?'
SAMPLE_DOWNLOAD_LINK += 'LeagueId=00&Season=2016-17&IsOnlyCurrentSeason=0'

PLAYER_DOWNLOAD_LINK = 'https://stats.nba.com/stats/playercareerstats?'
PLAYER_DOWNLOAD_LINK += 'PerMode=PerGame&PlayerID=%d'

# Choose the features to save and which to use for clustering
FEATURES_TO_GRAB = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'STL', 'BLK', 'PTS']
ACTIVE_FEATURES = ['FG_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'BLK', 'PTS']

# Define the bounds of the number of clusters to consider
MIN_CLUSTERS = 3
MAX_CLUSTERS = 5

# Define default ID to return when an error occurs with PLAYER_DOWNLOAD_LINK
ERROR_ID = 0


def loadData(link):
    # stats.nba.com only succesfully returns with a valid user agent
    user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML,'
    user_agent = user_agent + ' like Gecko) Chrome/61.0.3163.100 Safari/537.36'
    header = {'User-Agent': user_agent}
    # GET request using the requests module
    return requests.get(link, headers=header).json()

def downloadIds():
    link = SAMPLE_DOWNLOAD_LINK
    results = loadData(link)
    
    if not results:
        return None

    headers = results['resultSets'][0]['headers']
    data = results['resultSets'][0]['rowSet']
    df = pd.DataFrame(data, columns=headers)

    # Return a DataFrame with just the id and name for a given sample size
    filtered_df = df[['PERSON_ID', 'DISPLAY_LAST_COMMA_FIRST']].sample(SAMPLE_SIZE)
    return filtered_df.reset_index(drop=True)

def downloadPlayerData(id):
    id_link = PLAYER_DOWNLOAD_LINK %id
    id_response = loadData(id_link)

    if id_response is None:
        # Return None if unable to retrieve data
        return None

    player_headers = id_response['resultSets'][0]['headers']
    player_data = id_response['resultSets'][0]['rowSet']
    player_df = pd.DataFrame(player_data, columns=player_headers)
    return player_df

def filterData(id, first, last):
    player_df = downloadPlayerData(id)

    if player_df is None:
        # If it can't retrieve a specific player, return an empty DataFrame
        return pd.DataFrame()

    # Choice of features to retrieve by what seems reasonable to distinguish players
    # Average all of their NBA seasons to get a career average in these statistics
    avg = player_df[FEATURES_TO_GRAB].mean().to_frame().T
    # Add ID, first name, and last name to the DataFrame
    avg['player_id'] = id
    avg['first_name'] = first
    avg['last_name'] = last
    return avg

def fetchSummaryStatistics(players):
    final_df = pd.DataFrame()
    for index in range(players['PERSON_ID'].size):
        # Iterate through the sample, retrieve summary statistics
        # Compile a single DataFrame to be used for clustering
        id = players['PERSON_ID'][index]
        name = players['DISPLAY_LAST_COMMA_FIRST'][index].split(', ')
        # Retrieve name from "LAST, FIRST" (where len(name) == 1 => missing last name)
        last = name[0] if len(name) > 1 else ''
        first = name[1] if len(name) > 1 else name[0]
        final_df = final_df.append(filterData(id, first, last))

    # Return a single DataFrame with all of the player data
    return final_df

def findNumClusters(X):
    bounded_max = min(X.shape[0], MAX_CLUSTERS)
    range_n_clusters = range(MIN_CLUSTERS, bounded_max + 1)
    results = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(X, cluster_labels)
        results.append(silhouette_avg)

    # Return the number of clusters that corresponds to the best silhouette score 
    return range_n_clusters[np.argmax(results)]

def printFinal(df, n_clusters):
    final = {}
    filtered = df[['label', 'player_id', 'first_name', 'last_name']]
    for n in range(n_clusters):
        final[n] = filtered[filtered.label == n].drop(columns=['label']).to_dict('record')
    
    print(final)


if __name__ == '__main__':
    players = downloadIds()

    if players is None:
        error = 'Players could not be retrieved from: '
        raise Exception(error + SAMPLE_DOWNLOAD_LINK)

    df = fetchSummaryStatistics(players)

    if df.empty:
        error = 'Player summary statistics could not be retrieved from: '
        raise Exception(error + PLAYER_DOWNLOAD_LINK %ERROR_ID)

    # Get rid of players that are missing stats
    # Scale while preserving spareness of data and maximize silhouette score
    df = df.dropna().reset_index(drop=True)
    X = df[ACTIVE_FEATURES].values
    transformer = MaxAbsScaler().fit(X)
    X_scaled = transformer.transform(X)
    n_clusters = findNumClusters(X_scaled)

    # Fit the best number of clusters to the data
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_scaled)
    df['label'] = kmeans.labels_
    print(silhouette_score(X_scaled, kmeans.labels_))

    # Print out results
    printFinal(df, n_clusters)



