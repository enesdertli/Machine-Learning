from tokenize import maybe
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
"""
dir = 'S:\\Downloads\\data sets\\trainingData-20210813T103327Z-001\\trainingData'

categories = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62']
data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        img_letter = cv2.imread(imgpath,0)
        try:
            #* flatten is used to convert 2d array to 1d array
            img_letter = cv2.resize(img_letter, (50,50))
            image = np.array(img_letter).flatten()
            data.append([image, label])
        except Exception as e:
            pass

pick_in = open("data-letters.pickle",'wb')
pickle.dump(data, pick_in)
pick_in.close()
"""


#* Python pickle module is used for serializing and de-serializing a Python object structure. Any object in Python can be pickled so that it can be saved on disk. What pickle does is that it “serializes” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.
pick_in = open("data-letters.pickle",'rb')
data = pickle.load(pick_in)
pick_in.close()

#* The shuffle() method takes a sequence, like a list, and reorganize the order of the items.
random.shuffle(data)
features = []
labels = []
i = 0

for feature, label in data:
    features.append(feature)
    labels.append(label)
    print(i, " working")
    i = i +1
#* The train_test_split() function is used to split the dataset into training and testing sets. The test_size parameter specifies the ratio of the test set, which is set to 20% of the total dataset size.
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=20)
#* The SVC() function is used to create a Support Vector Classifier.
"""
model = SVC(C=1, gamma='auto', kernel='poly')
#* The fit() function is used to train the model using the training sets as parameters. The fit() method takes the training data as arguments, which can be one array in the case of unsupervised learning, or two arrays in the case of supervised learning. Note that the model is fitted using X and y , but the object holds no reference to X and y .
model.fit(x_train, y_train)
"""



pick = open('model-letters.sav', 'rb')
#pickle.dump(model, pick)
model = pickle.load(pick)
pick.close()


prediction = model.predict(x_test)
accuracy = model.score(x_test, y_test)

categories = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','1','2','3','4','5','6','7','8','9','0']


print("Accuracy: ",accuracy)
print("Prediction: ",categories[prediction[0]])

letter = x_test[0].reshape(50,50)
plt.imshow(letter, cmap='gray')
plt.show()

