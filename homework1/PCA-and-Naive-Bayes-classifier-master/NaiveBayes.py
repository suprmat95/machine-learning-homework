from PIL import Image
import numpy as np
import glob
from sklearn.naive_bayes import *
from sklearn.model_selection import *
from sklearn.decomposition import PCA

def load_image (infilename):

    img = Image.open(infilename)
    data = np.asarray(img, dtype=np.float64)
    data = data.ravel()
    return data

def my_trasform(data, X_t, components):

    return np.dot(data-X_t.mean_, components.T)

folders = ['dog' , 'guitar' , 'house' , 'person']
X = []
Y = []
for folder in folders:
    for infile in glob.glob(folder + "\*.jpg"):
        img = load_image(infile)
        X.append(img)
        Y.append(folder)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
clf = GaussianNB()
clf.fit(X_train, Y_train)
print("Predicted labels on test set:\n")
print(clf. predict(X_test))
print('\nAccuracy = ' + str(int(clf.score(X_test, Y_test)*100)) + '%')

X_std = (X - np.mean(X, axis=0))/(np.std(X, axis=0))
pca = PCA(4)
X_t = pca.fit(X_std)
pca2 = X_t.components_[:2, :]
trasformed2 = my_trasform(X_std, X_t, pca2)
X_train, X_test, Y_train, Y_test = train_test_split(trasformed2, Y, test_size=0.2)
clf.fit(X_train, Y_train)
print("Data projected onto their first two principal component\nSplitting, training and testing have been repeated")
print("\nPredicted label:\n")
print(clf. predict(X_test))
print('\nAccuracy = ' + str(int(clf.score(X_test, Y_test)*100)) + '%')

pca34 = X_t.components_[2:4, :]
trasformed34 = my_trasform(X_std, X_t, pca34)
X_train, X_test, Y_train, Y_test = train_test_split(trasformed34, Y, test_size=0.2)
clf.fit(X_train, Y_train)
print("Data projected onto their third and fourth principal component\nSplitting, training and testing have been repeated")
print("\nPredicted label:\n")
print(clf. predict(X_test))
print('\nAccuracy = ' + str(int(clf.score(X_test, Y_test)*100)) + '%')