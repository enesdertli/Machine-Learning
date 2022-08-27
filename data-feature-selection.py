from cgi import test
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

path = r"C:\Dosyalar\stuff\ders\csv files\diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path,names = names)
array = dataframe.values

#! Data Feature Selection Techniques
#? Univariate Selection
#* Univariate selection works by selecting the best features based on univariate statistical tests. We can use the chi-squared (chi²) statistical test for non-negative features to select 4 of the best features from the Pima Indians onset of diabetes dataset.

X = array[:,0:8]
Y = array[:,8]
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
set_printoptions(precision=2)
print(fit.scores_)
featured_data = fit.transform(X)
print("\nFeatured data:\n", featured_data[0:4])

#? Recursive Feature Elimination
#* Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.
"""
X = array[:,0:8]
Y = array[:,8]
model = LogisticRegression()
#! ERROR !
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Number of Features: %d")
print("Selected Features: %s")
print("Feature Ranking: %s")
"""

#? Principal Component Analysis
#* Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form. Generally this is called a data reduction technique. A property of PCA is that you can choose the number of dimensions or principal component in the transformed result.
#! ERROR !
"""
X = array[:,0:8]
Y = array[:,8]
pca = PCA(n_components=3)
fit = pca.fit(X)
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)
"""

#? Feature Importance
#* You can get the feature importance of each feature of your dataset by using the feature importance property of the model. As the name suggests, feature importance technique is used to choose the importance  features. It basically uses a trained supervised classifier to select features.
X = array[:,0:8]
Y = array[:,8]
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
#* From the output, we can observe that there are scores for each attribute. The higher the score, higher is the importance of that attribute