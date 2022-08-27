import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats as spstats
from sklearn.preprocessing import PolynomialFeatures

mpl.style.reload_library()
mpl.style.use('classic')
mpl.rcParams['figure.figsize'] = (6.0, 4.0)
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['figure.facecolor'] = (1,1,1,0)

poke_df = pd.read_csv("C:\Dosyalar\stuff\ders\csv files\Pokemon.csv")

print(poke_df.head())
print(poke_df[['HP','Attack','Defense']].head())
print(poke_df[['HP','Attack','Defense']].describe())

atk_def = poke_df[['Attack','Defense']]
print(atk_def.head())

#* The following code helps us build interaction features from these two features
pf = PolynomialFeatures(degree= 2, interaction_only = False, include_bias = False)
res = pf.fit_transform(atk_def)
print(res[:5])

#* We can clearly see from this output that we have a total of five features including the new interaction features. We can see the degree of each feature in the matrix, using the following snippet.
print(pd.DataFrame(pf.powers_, columns=['Attack_degree',"Defense_degree"]))

#* Now that we know what each feature actually represented from the degrees depicted, we can assign a name to each feature as follows to get the updated feature set.
print(intr_features = pd.DataFrame(res, columns=['Attack','Defense', 'Attack^2', 'Attack*Defense', 'Defense^2']))
