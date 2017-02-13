# With help/inspiration from: https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook
# and edX course: Microsoft: DAT210x Programming with Python for Data Science
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wheat = pd.read_csv('wheat.data') # Dataset available at https://archive.ics.uci.edu/ml/datasets/seeds
wheat.head()

wheat.drop('id',1,inplace=True)
# Histograms of a few features. We can look at their variance
s1 = wheat[['area','perimeter']]
s2 = wheat[['groove','asymmetry']]
s1.hist(alpha=0.75)
s2.hist(alpha=0.75)
plt.show()

# With a scatter plot we can look closer at these relationships
sns.FacetGrid(wheat, hue='wheat_type', size=5) \
            .map(plt.scatter, 'asymmetry', 'perimeter') \
            .add_legend()

# We take a look at the perimeter feature using a boxplot with datapoints jittered
#ax = sns.boxplot(x='wheat_type', y='perimeter', data=wheat)
#ax = sns.stripplot(x='wheat_type', y='perimeter', data=wheat, jitter=True, edgecolor='gray')

# Violing plot. Dense regions of the data are wider, sparse regions are thinner
plt.figure()
sns.violinplot(x='wheat_type', y='perimeter', data=wheat, size=6)

# Looking at univariate relations.
# kdeplot: creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(wheat, hue="wheat_type", size=6) \
   .map(sns.kdeplot, "perimeter") \
   .add_legend()
   
# A nice overview of relationships between different features. Area x Perimeter
# seem to have a really strong correlation
sns.pairplot(wheat, hue='wheat_type', size=2,diag_kind="kde")

# Andrews Curves helps visualize higher dimensionality, multivariate data, by 
# plotting each observation as a curve. The feature values act as coefficients of the curve.
from pandas.tools.plotting import andrews_curves
andrews_curves(wheat,'wheat_type')

# Parallel coordiantes let you view observations with more than three dimensions by tacking on
# additional parallel coordinates. Best use for limited number of features.
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(wheat, 'wheat_type')

# "Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature " ~ Ben Hamner, kaggle notebook
from pandas.tools.plotting import radviz
radviz(wheat, 'wheat_type')
