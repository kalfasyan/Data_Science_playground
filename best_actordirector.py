import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv('movie_metadata.csv')
actordirector = 'director_name' # actor_1_name
assert actordirector in ('actor_1_name','director_name'), 'Specify director_name or actor_1_name'
if actordirector == 'director_name':
    nrMovies = 10
elif actordirector == 'actor_1_name':
    nrMovies = 15

grouped = df.groupby(df[actordirector], as_index=False)
groupdf = grouped.imdb_score.agg([np.mean, np.std, len])    # mean, standard deviation, and nr-of-movies columns
groupdf['se'] = groupdf['std'] / np.sqrt(groupdf.len)       # standard error column
groupdf.dropna(axis=0, inplace=True)
groupdf = groupdf[groupdf.len>=nrMovies]                    # select actors/directors with more than nrMovies movies
groupdf.sort(['mean'],ascending=True,inplace=True)          # sorted by average imdb movie rating
groupdf.reset_index(inplace=True)
groupdf['names'] = groupdf.index


fig = groupdf.plot(kind='scatter', x='mean', y='names',yticks=range(len(groupdf)),xerr='se',figsize=(11,11))
fig.set_yticklabels(groupdf[actordirector]   , rotation=0)
plt.show()
