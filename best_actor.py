import pandas as pd
import numpy as np
#import ggplot 
from ggplot import *
import matplotlib.pyplot as plt

plt.style.use('ggplot')

df = pd.read_csv('movie_metadata.csv')

# I'm examining only 'first name' actors
# see alternative solution for 2nd/3rd actors
grouped = df.groupby(df['actor_1_name'], as_index=False)
groupdf = grouped.imdb_score.agg([np.mean, np.std, len])
groupdf.dropna(axis=0, inplace=True)
groupdf = groupdf[groupdf.len>=15]
groupdf.sort(['mean'],ascending=True,inplace=True)
groupdf.reset_index(inplace=True)
groupdf['names'] = groupdf.index
groupdf['se'] = groupdf['std'] / np.sqrt(groupdf.len)

fig = groupdf.plot(kind='scatter', x='mean', y='names',yticks=range(50),xerr='se',figsize=(12,12));
fig.set_yticklabels(groupdf.actor_1_name, rotation=0)
plt.show()

""" #Alternative solution with dictionary, correlations with imdb_score and Histogram of all ratings
# in case we want to check second/third names, we would concat all 3 actor-name 
# columns and take the unique from it. Then, create a similar dict as below.
#actornamesdf = pd.concat([df.actor_1_name,df.actor_2_name],axis=0)
actors = pd.unique(df.actor_1_name).tolist()
# Dict with names as keys and mean rating as values (for actors with more than 15 movies)
actoRatings = {i : np.mean(np.array(df[(df.actor_1_name==i)].imdb_score)) for i in actors if len(np.array(df[(df.actor_1_name==i)]))>15} 
import operator
sorted_x = sorted(actoRatings.items(), key=operator.itemgetter(1), reverse=True)
sorted_x[0:10]

def corr_score(feature):
    # To perform simple correlations with columns
    dfn = df[[feature,'imdb_score']]
    dfn = dfn[pd.notnull(dfn[feature])]
    print('correlation '+feature+' with imdb_score: ', np.corrcoef(dfn.ix[:,1],dfn.ix[:,0])[0,1])
    
corr_score('num_user_for_reviews')

# Histogram of all imdb ratings
df.imdb_score.hist(bins=10,range=(0,10))
"""
