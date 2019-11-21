
# coding: utf-8

# # A Least Squares Predictor for Fantasy Football
# 
# In Fantasy Football, contestants choose from a pool of available (American) football players to build a team.  Contestants' teams score points depending on how their chosen players performed in real-life.  The more points scored, the better!
# 
# There are literally hundreds of websites and blogs dedicated to predicting who will have a good game.  They use a variety of methodologies (including no methodology at all) to generate their predictions.  We will try to develop a predictor using Linear Least Squares that will answer the question: "Should I pick this player?"

# Bonus: This activity may help you with MP5, since you will be using similar data structures in that assignment.

# We'll import our standard packages, along with `pandas`, which is a `python` data analysis library.

# In[ ]:


import numpy as np
import numpy.linalg as la
import pandas as pd


# There are two data sets, `FF-data-2018.csv` and `FF-data-2019.csv` that were collected using scoring from the Yahoo Fantasy Football platform.  The 2018 data was collected from [here](http://rotoguru1.com/cgi-bin/fyday.pl?week=16&year=2018&game=yh&scsv=1).  You can choose other years going back to 2011 from a variety of platforms.
# 
# Let's read in the data and see what it looks like.

# In[ ]:


ff_2018 = pd.read_csv('FF-data-2018.csv')
ff_2018


# There are 6,355 data points which have a number of fields.  They are:
# - **Week**: The NFL season features 17 weeks of games, and each team plays 16 games in this time period.  This column tells you which week the player's game was.  I didn't include week 17, because many of the best players take that week off.
# 
# 
# - **Year**: Which year the game was played.  For this data set, all the year values are equal to 2018.
# 
# 
# - **GID**: A unique ID tag for each player.  We'll ignore this column.
# 
# 
# - **Name**: The actual name of the player.  In the case of defenses, the defense of the entire team is included, so in that case, this is the name of a city.
# 
# 
# - **Pos**: This is the position of the player.  The available choices are quarterback (QB), running back (RB), wide receiver (WR), tight end (TE), and defense (Def).
# 
# 
# - **Team**: An abbreviation that indicates which team the player belongs to.  Ryan Fitzpatrick was a member of the Tampa Bay Buccaneers, so his Team value is "tam".
# 
# 
# - **h/a**: Whether the player's game was played at home or on the road.  The possible values are 'h' (home) and 'a' (away).
# 
# 
# - **Oppt**: The opposing team that the player faced.  Ryan Fitzpatrick played against the New Orleans Saints in week 1, so his Oppt value is "nor".
# 
# 
# - **YH points**: The amount of points the player scored that week.  Ryan Fitzpatrick scored a whopping 42.28 points in week 1.
# 
# 
# - **YH salary**: On many Fantasy Football sites, you start with a certain budget, and select a team of players within the constraints of that budget.  Ryan Fitzpatrick only took 25.0 "dollars" of your available budget if you selected him on your team.  It gives an indication of how the platform judges the quality of a player.

# We can access the labels and put them in a list:

# In[ ]:


labels = list(ff_2018.columns)
print(labels)


# We can print out the available values of the positions for the data set by passing the key `Pos` as a string to the data set.

# In[ ]:


print(ff_2018['Pos'].values)


# To remove all the duplicates, we can call the function `numpy.unique` to access all distinct values.  (Just like every other time you use a new function, review the documentation of `numpy.unique`!  You can do so by running a cell with the following command: `np.unique?`)

# In[ ]:


positions = np.unique(ff_2018['Pos'])
print(positions)


# Since the positions in football are so different, we really want to focus on one at a time.  It would be very ambitious to try and create a general predictor for all positions.  Let's focus on quarterbacks first.  

# How can we extract all the data for quarterbacks?  We can find the rows in the dataframe that has position equal to  `QB`

# In[ ]:


POS = 'QB'
ff_2018['Pos'] == POS


# We will create another (smaller) dataframe that has the rows referring to the quarterback position.

# In[ ]:


df_POS = ff_2018[ff_2018['Pos'] == POS].copy()
df_POS.head()


# We can access the names of all the quarterbacks by referring to the columns `Name`

# In[ ]:


df_POS['Name']


# Linear Least Squares works with numerical data, not strings.  Eventually, we will want our predictive models to incorporate whether the player played at home or on the road, or how good their opponent was.  But the columns `h/a` and `Oppt` are strings:

# In[ ]:


df_POS['h/a']


# In[ ]:


df_POS['Oppt']


# At this point, we need to make decisions about what numerical values these should take.  For the home/away column: 
# 
# - let's make an array with the value +1.0 when the game is played at home, and -1.0 when the game is played away.
# 
# - store this array as another column in the pandas dataframe, with label `home_away`

# In[ ]:


df_POS['home_away'] = np.where(df_POS['h/a']=='a',-1,1)
df_POS


# For the opponents, we need some kind of information about how many points they give up to a position on average.  We have compiled that information in a separate file, called `team_rankings.py`.  Importing this file will give us access to a collection of dictionaries that provides this information.
# 
# After importing this file, the number `vs_2018[Pos][team]` will give us a relevant ranking.

# In[ ]:


from team_rankings import *  # asterik just means we import everything from that namespace


# We can take a look at the keys in the dictionary:

# In[ ]:


print( vs_2018.keys() )


# Note that the keys are just the player positions. Let's see the information for the key `QB` (we have been storing this string in the variable `POS`)

# In[ ]:


vs_2018[POS]


# In[ ]:


print(vs_2018[POS]['atl'])
print(vs_2018[POS]['buf'])


# There are 32 football teams in the NFL.  
# 
# The fact that `vs_2018['QB']['atl']` has the value 1.0, means that the Atlanta Falcons gave up the **most** points to quarterbacks on average in the 2018 season.  
# 
# Since `vs_2018['QB']['buf']` has the value 32.0, this means that the Buffalo Bills gave up the **least** points to quarterbacks on average in the 2018 season.
# 
# So, we would expect a better performance out of a quarterback if he is playing the Atlanta Falcons, compared to the Buffalo Bills. 

# The rankings can be very different for different positions:

# In[ ]:


print(vs_2018['RB']['atl'])
print(vs_2018['RB']['buf'])
print()
print(vs_2018['WR']['atl'])
print(vs_2018['WR']['buf'])
print()
print(vs_2018['TE']['atl'])
print(vs_2018['TE']['buf'])
print()
print(vs_2018['Def']['atl'])
print(vs_2018['Def']['buf'])
print()


# For the quarterback position (POS = 'QB'), convert the strings in the column `Oppt` into their corresponding numerical values using the dictionary `vs_2018`. Store this as another column of the pandas dataframe `oppt_rank`

# In[ ]:


def get_rank(x):
    return vs_2018[POS][x]

df_POS['oppt_rank'] = df_POS['Oppt'].apply(get_rank)
df_POS


# Now, players' names will be repeated in the array `names` for every game they played.  We will find it convenient to have another array collecting the names without these repeats.  We'll use `pandas.Series.unique` to do this.

# In[ ]:


unique_players = df_POS['Name'].unique()
len(unique_players)


# So 73 quarterbacks played in 2018.  But there are only 32 teams!  Who are all these people?

# In[ ]:


print(unique_players[7])
print(unique_players[72])


# I know who Tom Brady is, but I've never heard of Nate Sudfeld. Let's count how many times a players played a game.
# 
# We can use `groupby` to group players by Name, and then count the number of times each player appears:

# In[ ]:


df_POS.groupby('Name')['Name'].count()


# We want to add the frequency (game count) back to the original dataframe, and for that we will use transform to return an aligned index.

# In[ ]:


df_POS['game_count'] = df_POS.groupby('Name')['Name'].transform('count')
df_POS


# Note that Nate Sudfeld only played in 1 game in 2018.  He probably took over when the starter was injured, or when his team was involved in a lopsided game.  We probably want to remove his data, since it won't be very helpful.
# 
# Let's us create an array of the names of all the players that are relevant to our analysis. For that, we will exclude the names for all the players that participated in less than `min_games`.

# In[ ]:


min_games = 5
relevant_players =  df_POS[df_POS['game_count']>=min_games]['Name'].unique()
print(len(relevant_players))
relevant_players


# Now we only consider 43 quarterbacks playing in 2018.

# ### Let's put all of this together! 
# Write a function `prepare_data` that creates the dataframe `df_POS` for a given player position. The function also returns as an argument the list of relevant unique players.

# In[ ]:


def prepare_data(ff_data,POS,min_games):  
    # returns (new_df,relevant_players) as described above
    ...
    
    return(df_POS, relevant_players)


# Test out that your function works as expected:

# In[ ]:


df_test,players_test = prepare_data(ff_2018,'WR',3)
df_test


# # Simple Model - Last $n$ games
# 
# We'll start with a simple linear model. For now, we will keep using our example where we constructed a dataset for quarterbacks in the variable `df_POS`, along with `relevant_players`
# 
# The points scored in the previous $n$ games will be the only data considered when making a prediction.  Let's look at what the model would look like for only one player, say Andy Dalton, with $n = 3$.

# In[ ]:


pl = relevant_players[13]
pl_points = df_POS[df_POS['Name']==pl]['YH points'].values

print('Player:', pl)
print('Points:', pl_points)


# Andy Dalton played 11 games.  So we could try to build a model that predicted the points he scored in his 4th game, based on his first 3, and similarly try to predict the points he scored in the 5th games based on games 2,3, and 4.
# 
# I.e. a "local" least squares system might look something like
# 
# $$\mathbf{Ax}\cong \mathbf{b}$$
# 
# where
# 
# $$\mathbf{A} = \begin{pmatrix} 17.52 & 26.6 & 18.08\\ 26.6 & 18.08 & 25.78 \\ 18.08 & 25.78 & 13.92 \\
# 25.78 & 13.92 & 17.16 \\ 13.92 & 17.16 & 8.92 \\ 17.16 & 8.92 & 20.2 \\ 8.92 & 20.2 & 8.92 \\
# 20.2 & 8.92 & 19.34 \end{pmatrix}, \hspace{5mm} \mathbf{b}= \begin{pmatrix} 25.78 \\ 13.92 \\ 17.16 \\ 8.92 \\
# 20.2 \\ 8.92 \\ 19.34 \\ 9.1\end{pmatrix}$$

# This was with $n = 3$ games.  If instead, we base our "local" least squares on the previous $n = 4$ games, then our system would instead look like:
# 
# $$\mathbf{A} = \begin{pmatrix} 17.52 & 26.6 & 18.08 & 25.78\\ 26.6 & 18.08 & 25.78 & 13.92 \\ 
# 18.08 & 25.78 & 13.92 & 17.16\\ 25.78 & 13.92 & 17.16 & 8.92 \\ 13.92 & 17.16 & 8.92 & 20.2 \\ 
# 17.16 & 8.92 & 20.2 & 8.92\\ 8.92 & 20.2 & 8.92 & 19.34 \end{pmatrix},\hspace{4mm} \mathbf{b}= \begin{pmatrix} 13.92 \\ 17.16 \\ 8.92 \\
# 20.2 \\ 8.92 \\ 19.34 \\ 9.1\end{pmatrix} $$

# Write a function that generates this local system for a given (relevant) player.  Use the example above to debug your function (i.e., data for Andy Dalton)

# In[ ]:


def player_point_history(df, pl, n_games):   
    # df: dataframe
    # rel_player (string): name of a player
    # n_games (int): number of games used for the prediction
    ...
      
    return A,b

A,b = player_point_history(df_POS, relevant_players[13], 4) 
print(A)
print(b)


# Now, with this function, we can loop over the relevant players, generate their local systems, and "stack" them on top of each other to generate the global system.  We'll do this with $n = 3$

# In[ ]:


n_games = 3

# empty array for right hand side of size M x 1
pts_scored = np.array([])

# empty array for matrix of size M x n_games.  We had to reshape to size 0 x n_games to allow for "stacking" 
game_hist = np.array([]).reshape(0,n_games)

for pl in relevant_players:
    # generate local system
    a,c = player_point_history(df_POS,pl,n_games)
    
    # use numpy.append to append local system to global vector
    pts_scored = np.append(pts_scored,c)
    
    # use numpy.vstack (i.e. "vertical stack") to stack the global matrix and the local matrix
    game_hist = np.vstack((game_hist,a))
    
print(pts_scored.shape)
print(game_hist.shape)


# ### When should we start a player?
# 
# It would be an overly ambitious task to try to predict a players exact point total.  What we can do instead is set a "threshold".  I.e. if a player's points exceed this threshold, then we can deem them "startable".  If they don't exceed this threshold, then we should look choose a different player.
# 
# What threshold should we use?  That's debatable, but I've compiled the following dictionary based on additional data I collected from nfl.com.

# In[ ]:


start_threshold = {'QB': 19.3999, 'RB': 14.599, 'WR': 15.099, 'TE': 7.899, 'Def': 7.499}


# So, if a quarterback scores more than 19.3999, we declare them startable.  If a defense scores less than 7.499, then we should pick a different defense, etc.
# 
# We can finally set up our least squares system.  Set the matrix `A` to the variable `game_hist` defined above.  The components of the vector `b` should have a value of +1.0 if the corresponding component of `pts_scored` exceeds the threshold, and -1.0 if it lies below the threshold.  (I chose the thresholds so that it is impossible for the points to equal the threshold).

# Set up the right hand side vector, and solve the Linear Least Squares problem for $\mathbf{x}$.  You can use `numpy.linalg.lstsq` to compute the least-squares solution.  Then compute a numpy array `b_predict` that tests how this linear model performs on the data.

# In[ ]:


threshold = start_threshold[POS]
A = game_hist

b = ...
x = ...
b_predict = ...


# We can have the following situations:
# - The prediction tells you to start a player that ends up performing poorly (a "false positive")
# - The prediction tells you to exclude a player that ends up performing well (a "false negative")
# - The prediction tells you to start a player that ends up performing well (a correct prediction)

# Compute the number of false positives, false negatives, and correct prediction.  What percentage of each do we obtain on the data?

# The model is only correct 60.57% of the time.  However, it only return a "false positive" 3.39% of the time, which is very nice: if the model tells you to start a player, there's a good chance you will be happy with the results.

# Let's put it all together into a single function.  This will mostly be copying and pasting from above.  The function should return the variables `A`, `b`, `x`.  

# In[ ]:


def linear_predictor(ff_data, Pos, min_games, n_games, threshold):
    # clear
    ...
    
    return A, b, x


# We can call the routine for any position, and we can tweak the number of `min_games` and `n_games`.  You can also tweak the threshold.  Try changing the input variables and see how this affects model accuracy

# In[ ]:


Pos = 'WR'
min_games = 5
n_games = 3
threshold = start_threshold[Pos]

A, b, x = linear_predictor(ff_2018, Pos, min_games, n_games, threshold)

b_predict = ...


# Notice we didn't make use of the fact that a player is playing on home or on the road, or the ranking of the opponent.  Let's try to enrich the features used in this problem to include this data.  Let's go back to Andy Dalton:

# In[ ]:


pl = relevant_players[13]
pl_points = df_POS[df_POS['Name']==pl]['YH points'].values
pl_home_away = df_POS[df_POS['Name']==pl]['home_away'].values
pl_oppt_rank = df_POS[df_POS['Name']==pl]['oppt_rank'].values

print('Player:', pl)
print('Points:', pl_points)
print('Location:', pl_home_away)
print('Opp Rank:', pl_oppt_rank)


# When $n = 3$ we had the following system when we only took previous games played:
# 
# $$\mathbf{A} = \begin{pmatrix} 17.52 & 26.6 & 18.08\\ 26.6 & 18.08 & 25.78 \\ 18.08 & 25.78 & 13.92 \\
# 25.78 & 13.92 & 17.16 \\ 13.92 & 17.16 & 8.92 \\ 17.16 & 8.92 & 20.2 \\ 8.92 & 20.2 & 8.92 \\
# 20.2 & 8.92 & 19.34 \end{pmatrix}, \hspace{5mm} \mathbf{b}= \begin{pmatrix} 25.78 \\ 13.92 \\ 17.16 \\ 8.92 \\
# 20.2 \\ 8.92 \\ 19.34 \\ 9.1\end{pmatrix}$$
# 
# With the location and opponent data, it should now look like this:
# 
# $$\mathbf{A} = \begin{pmatrix} 17.52 & 26.6 & 18.08 & -1 & 1\\ 26.6 & 18.08 & 25.78 & 1 & 10 \\ 18.08 & 25.78 & 13.92 & 1 & 17\\ 25.78 & 13.92 & 17.16 & -1 & 5\\ 13.92 & 17.16 & 8.92 & 1 & 4\\ 17.16 & 8.92 & 20.2 & 1 & 2\\ 8.92 & 20.2 & 8.92 & -1 & 29 \\
# 20.2 & 8.92 & 19.34  & 1 & 13\end{pmatrix}, \hspace{5mm} \mathbf{b}= \begin{pmatrix} 25.78 \\ 13.92 \\ 17.16 \\ 8.92 \\
# 20.2 \\ 8.92 \\ 19.34 \\ 9.1\end{pmatrix}$$

# Create an enriched linear regression, by adding these two extra columns to the matrix $\mathbf{A}$.  The routine should return `A` with the two added columns.  It should also return the right hand side `b` and least-squares solution `x`.

# In[ ]:


def linear_predictor_enriched(ff_data, Pos, min_games, n_games, threshold):
    ...
    
    return A, b, x


# You should see that the enriched version is considerably better for running backs, with our standard inputs:

# In[ ]:


Pos = 'RB'
min_games = 5
n_games = 3
threshold = start_threshold[Pos]

A, b, x  = linear_predictor(ff_2018, Pos, min_games, n_games, threshold)

b_predict = ...

print('Standard Model')
print('Fraction of false negatives:    ', ...)
print('Fraction of false positives:    ', ...)
print('Fraction of correct predictions:', ...)
print()

A, b, x = linear_predictor_enriched(ff_2018, Pos, min_games, n_games, threshold)

b_predict = ...

print('Enriched Model')
print('Fraction of false negatives:    ', ...)
print('Fraction of false positives:    ', ...)
print('Fraction of correct predictions:', ...)
print()


# But you'll find it's not very effective for quarterbacks:

# In[ ]:


Pos = 'QB'
min_games = 5
n_games = 3
threshold = start_threshold[Pos]

A, b, x  = linear_predictor(ff_2018, Pos, min_games, n_games, threshold)

b_predict = ...

print('Standard Model')
print('Fraction of false negatives:    ', ...)
print('Fraction of false positives:    ', ...)
print('Fraction of correct predictions:', ...)
print()

A, b, x = linear_predictor_enriched(ff_2018, Pos, min_games, n_games, threshold)

b_predict = ...

print('Enriched Model')
print('Fraction of false negatives:    ', ...)
print('Fraction of false positives:    ', ...)
print('Fraction of correct predictions:', ...)
print()


# The number of false positives has shot up dramatically.  Despite the (slightly) better accuracy, I would probably avoid this one.
# 
# It seems that running backs are more "matchup-dependent" than quarterbacks.  That is, where they are playing and how good the other team is are bigger factors in their performance compared to quarterbacks.

# # Validation set
# Of course, you never want to conclude anything about your model based on the data you used to construct it.  You should validate its accuracy on a different data set.  We can do so on this years fantasy football data.  We can also select the optimal **hyperparameters** (a fancy word for parameters) based on this validation set.
# 
# Some questions to ask as you test the model on the validation set:
# 
# - Should we include the home/away and opponent data or not?
# - Is our decision to exclude players that have played less than 5 games a good one?  Should we bump that number up to 7 games?  Or down to 3?
# - How many games should we include in our history?  Is 3 games really the best choice?  What about 5?  What about just the last game?
# 
# I.e. the inclusion of the extra data, the minimum number of games, and the history length are the **hyperparameters** for this model.

# In[ ]:


ff_2019 = pd.read_csv('FF-data-2019.csv')

# position
Pos = 'TE'

# these are your hyperparameters
min_games = 6
n_games = 3
enriched = True

# build model on 2018 data and retrieve least squares solution x
if enriched:
    OUT_2018 = linear_predictor_enriched(ff_2018, Pos, min_games,n_games,threshold)
    x = OUT_2018[2]
else:
    OUT_2018 = linear_predictor(ff_2018, Pos, min_games,n_games,threshold)
    x = OUT_2018[2]
    

# retrieve Data matrix A and outcomes vector b using 2019 data
if enriched:
    OUT_2019 = linear_predictor_enriched(ff_2019, Pos, min_games,n_games,threshold)
    A,b = OUT_2019[0], OUT_2019[1]
else:
    OUT_2019 = linear_predictor(ff_2019, Pos, min_games,n_games,threshold)
    A,b = OUT_2019[0], OUT_2019[1]
    
# assess model
b_predict = ...

print('Enriched Model')
print('Fraction of false negatives:    ', ...)
print('Fraction of false positives:    ', ...)
print('Fraction of correct predictions:', ...)
print()

