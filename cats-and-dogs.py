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
dir = 'S:\\Downloads\\kagglecatsanddogs_5340\\PetImages'

categories = ['Cat', 'Dog']
data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        pet_img = cv2.imread(imgpath,0)
        try:
            #* flatten is used to convert 2d array to 1d array
            pet_img = cv2.resize(pet_img, (50,50))
            image = np.array(pet_img).flatten()
            data.append([image, label])
        except Exception as e:
            pass

pick_in = open("data.pickle",'wb')
pickle.dump(data, pick_in)
pick_in.close()
"""

#* Python pickle module is used for serializing and de-serializing a Python object structure. Any object in Python can be pickled so that it can be saved on disk. What pickle does is that it “serializes” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.
pick_in = open("data.pickle",'rb')
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
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.01)
#* The SVC() function is used to create a Support Vector Classifier.
#model = SVC(C=1, gamma='auto', kernel='poly')
#* The fit() function is used to train the model using the training sets as parameters. The fit() method takes the training data as arguments, which can be one array in the case of unsupervised learning, or two arrays in the case of supervised learning. Note that the model is fitted using X and y , but the object holds no reference to X and y .
#model.fit(x_train, y_train)



pick = open('model44.saved', 'rb')
#pickle.dump(model, pick)
model = pickle.load(pick)
pick.close()

prediction = model.predict(x_test)
accuracy = model.score(x_test, y_test)

categories = ['Cat', 'Dog']


print("Accuracy: ",accuracy)
print("Prediction: ",categories[prediction[0]])

my_pet = x_test[0].reshape(50,50)
plt.imshow(my_pet, cmap='gray')
plt.show()
