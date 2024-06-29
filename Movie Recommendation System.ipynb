#**Movie Recommendation System**

#---


#**objective**

#---
#**Recommendation system** is a system that seeks to predict or filter preferences according to the user's choices. Recommendation systems are used in avariety of areas such as movies, news, book,research articles, search queries, social tags and products in general. Recommender systems produce a list of recommendations in any of the two ways -

  

#1. **Collaborative filtering** : Collaborative filetring approaches build a model from on the user's past behaviour(i.e. items purchased or searched by the user) as well as similae decisions made by other users. This model is then used to predict items(or ratings for items)that users may have ineterest in.
#2. **Content-based filtering** : Content-based filteringapproaches uses a series of discrete characteristics of an item in order to recommend additional items with similar properties. Content-based filteringmethods are totally based on a description of the item and a profile of the user's preferences. It recommends items based on the user's past preferences. let's develop a basic recommendation system using pyhton and pandas.

 # Let's develop a basic recommendation system by suggestng items that are most similar to a particular, in this case, movies. It just tells what movies/items are most similar to the user's movie choice.




##**Data Source**


#https://github.com/YBI-Foundation/Dataset/raw/main/Movies%20Recommendation.csv




###**Import Library**

import pandas as pd
import numpy as np

###**Import dataset**

df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Movies%20Recommendation.csv')


df.head()

df.info()

###**Describe Data and Data Visualization**

df.describe()

###**Data Preprocessing**

df.shape

df.columns

###**Define Target Variables(y) and Feature variables(X)**

df_features = df[['Movie_Budget','Movie_Popularity','Movie_Revenue', 'Movie_Vote', 'Movie_Vote_Count' ]]

#Selected these 5 existing features to recommend movies. It can vary from project to project with the person's interest and requirements.

df_features.shape

df_features

y=df['Movie_ID']

X=df[['Movie_Budget','Movie_Popularity','Movie_Revenue', 'Movie_Vote', 'Movie_Vote_Count']]

y.shape

X.shape

###**Train Test Split**

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.7, random_state=2529)

X_train, X_test, y_train, y_test

X_train.shape, X_test.shape, y_train.shape, y_test.shape

###**Modeling**

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

model.intercept_

model.coef_

###**Prediction**

y_pred = model.predict(X_test)

X_test

###**Model Evaluation**

from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(y_test, y_pred)

####**Get Similarity Scoe using Cosine Similarity**

from sklearn.metrics.pairwise import cosine_similarity

Similarity_Score = cosine_similarity(X)

Similarity_Score

Similarity_Score.shape

####**Get Movie Name as Input from User and Validate the Closest Spelling**

Favourite_Movie_Name = input('Enter your favourite movie name:')

All_Movies_Title_List = df['Movie_Title'].tolist()

import difflib

Movie_Recommendation = difflib.get_close_matches(Favourite_Movie_Name, All_Movies_Title_List)
print(Movie_Recommendation)

Close_Match = Movie_Recommendation[0]
print(Close_Match)

Index_of_Close_Match_Movie = df[df.Movie_Title == Close_Match]['Movie_ID'].values[0]
print(Index_of_Close_Match_Movie)

Recommendation_Score = list(enumerate(Similarity_Score[Index_of_Close_Match_Movie]))
print(Recommendation_Score)

len(Recommendation_Score)

####**Get All Movies Sort Based on Recommendation Score wrt Favourite Movie**

#Sorting the Movies based on their Similarity_Score
Sorted_Similar_Movies = sorted(Recommendation_Score, key = lambda x:x[1], reverse = True)
print(Sorted_Similar_Movies)

#Print the Name of the Movie Based on the Index
print('Top 30 Movies Suggested for you : /n')
i=1
for movie in Sorted_Similar_Movies:
 index = movie[0]
 title_from_index = df[df.index==index]['Movie_Title'].values[0]
 if(i<31):
  print(i , '.' , title_from_index)
  i+=1

####**Top 10 Movie Recommendation System**

Movie_Name = input('Enter your favourite movie name :')
list_of_all_titles = df['Movie_Title'].tolist()
Find_Close_Match = difflib.get_close_matches(Movie_Name, list_of_all_titles)
if Find_Close_Match :
 Close_Match = Find_Close_Match[0]
Index_of_Movie = df[df.Movie_Title == Close_Match]['Movie_ID'].values[0]
Recommendation_Score = list(enumerate(Similarity_Score[Index_of_Close_Match_Movie]))
Sorted_Similar_Movies = sorted(Recommendation_Score, key = lambda x:x[1], reverse = True)
print('Top 10 Movies Suggested for you : \n')
i=1
for movie in Sorted_Similar_Movies :
  index = movie[0]
  title_from_index = df.iloc[index]['Movie_Title']
  if(i<11):
    print(i , '.' , title_from_index)
    i+=1

##**Explanation**

#---



#This code implements a movie recommendation system based on a user's favorite movie. The goal is to suggest movies that are similar in characteristics to the user's input.
