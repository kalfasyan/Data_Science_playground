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
sns.violinplot(x='wheat_type', y='perimeter', data=wheat, size=6)

# Looking at univariate relations.
# kdeplot: creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(wheat, hue="wheat_type", size=6) \
   .map(sns.kdeplot, "perimeter") \
   .add_legend()
   
# A nice overview of relationships between different features. Area x Perimeter
# seem to have a really strong correlation
sns.pairplot(wheat, hue='wheat_type', size=2,diag_kind="kde")
