from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from pandas.plotting import scatter_matrix

path = r"C:\Dosyalar\stuff\ders\csv files\diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)

#! Statistic part
#* Reviewing correlation between attributes
correlations = data.corr(method='pearson')
print(correlations)

#* Reviewing skew of attribute distribution
print("///////////")
print(data.skew())

#! Data visualization techniques --> univariate plots --> histogram, density plots, box plots

#? Histogram
#* could say age, pedi and test attribute may have exponential distribution while mass and plas have Gaussian distribution
data.hist()
#pyplot.show()

#? Density plots
data.plot(kind="density", subplots = True, layout=(3,3), sharex = False)
#pyplot.show()
#* Density plots. From the above output, the difference between Density plots and Histograms can be easily understood.

#? Box plots
data.plot(kind = "box", subplots = True, layout = (3,3), sharex = False, sharey = False)
#pyplot.show()
#* Box and whisker plots From the above plot of attribute’s distribution, it can be observed that age, test and skin appear skewed towards smaller values.

#! Data visualization techniques --> Multivariate Plots --> Correlation Matrix Plots
#? Correlation Matrix Plot
correlations = data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
#pyplot.show()
#* From the above output of correlation matrix, we can see that it is symmetrical i.e. the bottom left is same as the top right. It is also observed that each variable is positively correlated with each other.

#? Scatter Matrix Plot
scatter_matrix(data)
pyplot.show()


