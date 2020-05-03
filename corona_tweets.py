#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:26:32 2020

@author: krishanagyanwali
"""

import tweepy as tw
import csv
import twitter_keys as tk

auth = tw.OAuthHandler(tk.consumer_key, tk.consumer_secret)
auth.set_access_token(tk.access_token_key, tk.access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Open/create a file to append data to
csvFile = open('coronavirus3.csv', 'a')

coronavirus_tweets = csvWriter = csv.writer(csvFile)



# Define the search term and the date_since date as variables
search_words = "#coronavirus"
date_since = "2020-03-15"

# Collect tweets
for tweet in tw.Cursor(api.search,
                           q = "coronavirus",
                           since = "2020-01-22",
                           until = "2020-03-22",
                           lang = "en").items():

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    print (tweet.created_at, tweet.text)
csvFile.close()
