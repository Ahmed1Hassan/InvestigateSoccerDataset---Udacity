#!/usr/bin/env python
# coding: utf-8

# # Project: European Soccer Database Analysis
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > Welcome to Soccer Data Analysis Project, in this project, we will analyze European Soccer Database (made by Hugo Mathien). This soccer database comes from Kaggle and is well suited for data analysis and machine learning. It contains data for soccer matches, players, and teams from several European countries from 2008 to 2016.
# 
# >### Data Analysis is performed to answer the following questions:
# (1) What teams improved the most over the time period? \
# (2) Which players had the most penalties? \
# (3) What team attributes lead to the most victories?
# 
# >N.B. The database is stored in a SQLite database. To access the database files, use software like DB Browser.

# In[1]:


# Packages Importing
import sqlite3 as sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### Connect the Project to the Database

# In[2]:


# Establish connection to database, and load the tables.
conn=sql.connect('soccer_database.sqlite')


# In[3]:


# Find out the tables in the databse
df_tables=pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
df_tables


# ### Select The Needed Tables From The Database

# In[4]:


# First: Match Table
df_match=pd.read_sql_query('SELECT * FROM match',conn)


# #### General Properties of Match Table

# In[5]:


# View first Data of Match Table
df_match.head()


# In[6]:


# Count Duplicates in Match Table
df_match.duplicated().sum()


# In[7]:


# Second: Team Table
df_team=pd.read_sql_query('SELECT * FROM team',conn)


# #### General Properties of Team Table

# In[8]:


# View first Data of Team Table
df_team.head()


# In[9]:


# Count Duplicates in Team Table
df_team.duplicated().sum()


# In[10]:


# Check for NaN values count
df_team.isnull().sum()


# ##### Note: team_fifa_api_id Column contains NaN Values but it is not important for the project . To Clean the Team Table, I will drop this column.

# In[11]:


#Third: Team Attributes Table
df_team_attributes=pd.read_sql_query('SELECT * FROM team_attributes',conn)


# #### General Properties of Team Table

# In[12]:


# View first Data of Team Attributes Table
df_team_attributes.head()


# In[13]:


# Count Duplicates in Team Attributes Table
df_team_attributes.duplicated().sum()


# In[14]:


# Check for missing data
df_team_attributes.info()


# In[15]:


#check for NaN values count
df_team_attributes.isnull().sum()


# ##### Note: buildUpPlayDribbling Column contains over 66% NaN Values. To Clean the Team Attributes Table, I will drop this column.

# In[16]:


# Fourth: Player Table
df_player=pd.read_sql_query('SELECT * FROM player', conn)


# #### General Properties of Player Table

# In[17]:


# View first Data of Player Table
df_player.head()


# In[18]:


# Count Duplicates in Player Table
df_player.duplicated().sum()


# In[19]:


# Check for NaN values count
df_player.isnull().sum()


# In[20]:


# Fifth: Player Attributes Table
df_player_attributes=pd.read_sql_query('SELECT * FROM player_attributes;', conn)


# #### General Properties of Player Attributes Table

# In[21]:


# View first Data of Player Attributes Table
df_player_attributes.head()


# In[22]:


# Check for NaN values count
df_player_attributes.isnull().sum()


# In[23]:


# Count Duplicates in Team Table
df_player_attributes.duplicated().sum()


# In[24]:


# Check for NaN values count
df_player_attributes.isnull().sum()


# In[25]:


# Count number of rows that contains missing values
df_player_attributes['null_count']=df_player_attributes.isnull().sum(axis=1)
(df_player_attributes['null_count']!=0).sum()


# In[26]:


# Calculate their percentage of number of rows that contain NaN
(df_player_attributes['null_count']!=0).sum()/df_player_attributes.shape[0]


# #### To Clean Player Attributes Table, Rows with NaN Values will be dropped

# ### Data Cleaning

# #### Team Attributes Table: Drop 'buildUpPlayDribbling' Column

# In[27]:


# Drop buildUpPlayDribbling Column
df_team_attributes.drop(columns='buildUpPlayDribbling', inplace=True)


# #### Player Attributes Table: Drop rows containing NaN then merge with player table

# In[28]:


# Merge player and player_attributes
df_player_data=pd.merge(df_player_attributes, df_player, on=['player_api_id', 'player_fifa_api_id'])


# #### Team Table: Drop 'team_fifa_api_id' Column

# In[29]:


# Drop team_fifa_api_id column out of Team Table
df_team.drop(columns='team_fifa_api_id', inplace=True)


# #### Cleaing Match Table:
# ##### (1) Extract First Columns
# ##### (2) Calculate Each Team Points in Each Match
# ##### (3) Aggregate Team Points to Points per Game of Individual Year
# ##### (4) Save them to New Dataframe team_data
# ##### (5) Merge team_attributes to team_data for analysis

# In[30]:


df_match=df_match.copy()


# In[31]:


# Extract First Match Columns
match=df_match[['id', 'country_id', 'league_id', 'season', 'stage', 'date','match_api_id', 'home_team_api_id', 'away_team_api_id','home_team_goal', 'away_team_goal']]


# In[32]:


# Calculate Each Team Points in Each Match
pd.options.mode.chained_assignment = None
match['home_points']= (match.home_team_goal > match.away_team_goal)*3 + (match.home_team_goal == match.away_team_goal)*1
match['away_points']= (match.home_team_goal < match.away_team_goal)*3 + (match.home_team_goal == match.away_team_goal)*1
match.head()


# In[33]:


# Split the match table into home dataframe and away dataframe, then append them into a single dataframe
homedf=match[['home_team_api_id', 'date','home_team_goal','home_points']]
homedf.columns=['team_api_id', 'date', 'goal','points']
awaydf=match[['away_team_api_id', 'date','away_team_goal','away_points']]
awaydf.columns=['team_api_id', 'date', 'goal','points']
df1=pd.DataFrame(homedf.append(awaydf))


# In[34]:


# Create a year column
df1['date']=pd.to_datetime(df1['date'])
df1['year']=df1['date'].dt.year


# In[35]:


# Create a new dataframe team_data, that aggregates points into point per game for each year
team_data=pd.DataFrame()
team_data['total_goals']=df1.groupby(['team_api_id','year'])['goal'].sum()
team_data['num_of_games']=df1.groupby(['team_api_id','year']).goal.count()
team_data['total_points']=df1.groupby(['team_api_id','year']).points.sum()
team_data['points_per_game']=team_data['total_points'] / team_data['num_of_games']
team_data.reset_index(inplace=True)


# In[36]:


#merge with the table team to include team long name, and drop other unnecessary columns
team=pd.read_sql_query("SELECT * FROM team;", conn)
team_data=pd.merge(team_data, team, on='team_api_id')
team_data.drop(columns=['id', 'team_fifa_api_id','team_short_name'], inplace=True)


# ##### Only Teams that have been playing every year since 2010, and played at least 5 years since 2010 will be considered. All The Rows Before 2010 will be dropped. 

# In[37]:


# Drop the rows where the year is before 2010
team_data=team_data[team_data.year>2009]
team_data.head()


# In[38]:


# Create a dataframe that counts how many year each team played
num_years=team_data.groupby('team_api_id').year.count()
num_years=pd.DataFrame(num_years)
num_years.columns=['num_years']
num_years.reset_index(inplace=True)
num_years.head()


# In[39]:


# Merge the team_per and num_year dataframes
team_data=pd.merge(team_data, num_years, on='team_api_id')


# In[40]:


# Filter the dataframe to select only teams who has played at least 5 years 
team_data=team_data.query('num_years>4')


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 

# ### Question #1: Which Team Improved the most over the year?

# In[41]:


# Calculate Slope of Points Per Game vs. Year under name team_improvement 
team_improvement=team_data.groupby('team_long_name').apply(lambda v: stats.linregress(v.year,v.points_per_game)[0])
# Display 5 Most Improved Teams
team_improvement.sort_values(axis=0, ascending=False).head(5)


# In[42]:


# Histogram of Team Imporvements
team_improvement.hist(bins=50)
plt.xlabel('Improvement Points per Game & Year')
plt.ylabel('Number Of Teams')
plt.title('Team Improvements')


# In[43]:


# Visualization Plot of the Points Per Game vs. Year for the most improved teams
x1=team_data.query('team_long_name=="Southampton"')['year']
y1=team_data.query('team_long_name=="Southampton"')['points_per_game']
x2=team_data.query('team_long_name=="Dundee FC"')['year']
y2=team_data.query('team_long_name=="Dundee FC"')['points_per_game']
x3=team_data.query('team_long_name=="Juventus"')['year']
y3=team_data.query('team_long_name=="Juventus"')['points_per_game']
plt.plot(x1,y1, label='Southampton')
plt.plot(x2,y2, label="Dundee FC")
plt.plot(x3,y3, label="Juventus")
plt.legend()
plt.xlabel('Year')
plt.ylabel('Points per Game')
plt.title("The Most Improved Teams")


# ### Question #2: Which Player has the most penalties?

# In[44]:


# Sorting Values by Penalties Column
df_penalties=df_player_data.sort_values(by=['penalties'], ascending=False)


# In[45]:


### Display Players with Most Penalties
sns.barplot(y="player_name", x="penalties", data=df_penalties[:25])
plt.xlabel('No. of Penalties')
plt.ylabel('Player Name')
plt.title("The Most Players with Penalties");
plt.xlim(90,100)


# ### Question #3: Which Team Attributes Lead To Most Victories?

# In[46]:


# Create Year Column in Team Attributes Table
df_team_attributes['year']=pd.to_datetime(df_team_attributes['date']).dt.year


# In[47]:


# Merge team_data and Team Attributes on team_api_id & Year
team_data=pd.merge(team_data, df_team_attributes, on=['team_api_id', 'year'])


# In[48]:


# View Data Information
team_data.info()


# In[49]:


# Create Dataframe for Quantitative Attributes Only as Categorical Attributes are measured by the Quantitative Ones so no need to compare them.
df_quant_attributes=team_data[['team_api_id', 'team_long_name', 'year', 'points_per_game','buildUpPlaySpeed','buildUpPlayPassing','chanceCreationPassing','chanceCreationCrossing','chanceCreationShooting','defencePressure','defenceAggression','defenceTeamWidth']]


# In[50]:


# Calculate the Correlation between Quantitative Attributes & Points Per Game.
df_quant_attributes.corr()


# In[51]:


# Calculate Kendall's Correlation between Points Per Game & Defence Pressure
stats.kendalltau(df_quant_attributes.points_per_game, df_quant_attributes.defencePressure)


# In[52]:


# Calculate Kendall's Correlation between Points Per Game & Defence Aggression
stats.kendalltau(df_quant_attributes.points_per_game, df_quant_attributes.defenceAggression)


# In[53]:


# Calculate Kendall's Correlation between Points Per Game & Chance Creation Shooting
stats.kendalltau(df_quant_attributes.points_per_game, df_quant_attributes.chanceCreationShooting)


# ##### Top 3 Team Attributes that are best positively correlated with Points Per Game (P Value = 0):
# 1. Defence Pressure
# 2. Defence Aggression 
# 3. Chance Creation Shooting

# In[54]:


# Close Connection TO SQLite Database
conn.close()


# <a id='conclusions'></a>
# ## Conclusions
# 
# ### Limitations
# > (1) Missing Data (E.x: BuildUpDribbling Column has NaN values over 66% of its data) \
# > (2) Insufficient Data for better calculation results
# 
# ### Final Answers 
# #### (1) Which Team Improved The Most Over The Years?   
#    > Southampton Team
# #### (2) Which Player has The Most Penalties?
#    > Rickie Lambert
# #### (3) Which Team Attributes Lead To Most Victories?
#    > 1. Defence Pressure
#    > 2. Defence Aggression
#    > 3. Chance Creation Shooting

# ## Acknowledgements
# >- Udacity's Data Analysis Nanodegree Program
# >- Python & Pandas Documentation
# >- StackOverFlow Website

# In[ ]:




