from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler


path = r"C:\Dosyalar\stuff\ders\csv files\diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values

#! Data Pre-processing Techniques
#? Scalling
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_rescaled = data_scaler.fit_transform(array)
set_printoptions(precision=1)
print("\nScaled data:\n", data_rescaled[0:10])
#* Most probably our dataset comprises of the attributes with varying scale, but we cannot provide such data to ML algorithm hence it requires rescaling. Data rescaling makes sure that attributes are at same scale. Generally, attributes are rescaled into the range of 0 and 1. ML algorithms like gradient descent and k-Nearest Neighbors requires scaled data. 

#? Normalization  (L1 Normalization and L2 Normalization) 
#* Another useful data preprocessing technique is Normalization. This is used to rescale each row of data to have a length of 1. It is mainly useful in Sparse dataset where we have lots of zeros.
#? L1 Normalization
Data_normalizer = Normalizer(norm="l1").fit(array)
Data_normalized = Data_normalizer.transform(array)
set_printoptions(precision=2)
print("\nNormalized data:\n", Data_normalized[0:3])

#? L2 Normalization
Data_normalizer = Normalizer(norm="l2").fit(array)
Data_normalized = Data_normalizer.transform(array)
set_printoptions(precision=2)
print("\nNormalized data:\n", Data_normalized[0:3])

#? Binarization
#* Binarization is used to convert continuous data to binary data. It is useful in classification problems where we have to classify data into two classes.
binarizer = Binarizer(threshold=0.5).fit(array)
Data_binarized = binarizer.transform(array)
print("\nBinary data:\n", Data_binarized[0:5])

#? Standardization
#* Standardization is used to rescale data to have a mean of 0 and standard deviation of 1. It is useful in regression problems where we have to predict the value of a continuous variable.
data_scaler = StandardScaler().fit(array)
data_rescaled = data_scaler.transform(array)
set_printoptions(precision=2)
print("\nStandardized data:\n", data_rescaled[0:5])

#? Data Labeling
#* Data Labeling is used to convert continuous data to categorical data. It is useful in classification problems where we have to classify data into two classes. Most of the sklearn functions expect that the data with number labels rather than word labels. Hence, we need to convert such labels into number labels. This process is called 
input_labels = ["red", "black", "red", "green", "black", "yellow", "white"]
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
test_labels = ["green", "red", "black"]
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("Encoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))