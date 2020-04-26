import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import *

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def print_graph(titlee, clf):
    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(titlee)
    plt.show()
    plt.close()

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4)


#Linear SVM
scores = []
max_score = 0
Cs = []
for n in range(-3, 4):
    C = pow(10, n)
    Cs.append(C)
    model = svm.SVC(kernel='linear', C=C)
    clf = model.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    if score > max_score:
        max_score = score
        max_clf = clf
        max_C = C
    scores.append(score)
    print('Score C = ' + str(C) + '  score = ' + str(int(score * 100)) + '%')
    title = ('Decision surface of linear SVC ' + '(C = ' + str(C) + ')')
    print_graph(title, clf)

score = max_clf.score(X_test, y_test)
print('Score on test set: C = ' + str(max_C) + '  score = ' + str(int(score * 100)) + '%')

title = ('Decision surface on the best of linear SVC ' + '(C = ' + str(max_C) + ')')
print_graph(title, max_clf)

plt.plot(Cs, scores)
plt.xscale('log')
plt.xlabel('Accuracy')
plt.ylabel('Value of C')
plt.show()
plt.close()

#RBF kernel
max_score = 0
for n in range(-3, 4):
    C = pow(10, n)
    Cs.append(C)
    model = svm.SVC(kernel='rbf', C=C, gamma='auto')
    clf = model.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    if score >= max_score:
        max_score = score
        max_clf = clf
        max_C = C
    scores.append(score)
    print('Score C = ' + str(C) + '  score = ' + str(int(score * 100)) + '%')
    title = ('Decision surface of RBF-kernel' + '(C = ' + str(C) + ')')
    print_graph(title, clf)

score = max_clf.score(X_test, y_test)
print('Score on test set with the best RBF-kernel: C = ' + str(max_C) + '  score = ' + str(int(score * 100)) + '%')
title = ('Decision surface on the best of RBF-kernel ' + '(C = ' + str(max_C) + ')')
print_graph(title, max_clf)


rows = []
for i in range(-9, 2):
    gamma = pow(10, i)
    rows.append(gamma)
columns = []
for i in range(-3, 6):
    C = pow(10, i)
    columns.append(C)

print('Grid search of the best parameters for an RBF kernel:\n\n')
max_score = 0
for C in columns:
    row = []
    for gamma in rows:
        model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        clf = model.fit(X_train, y_train)
        score = clf.score(X_val, y_val)
        if score >= max_score:
            max_score = score
            max_clf = clf
            max_C = C
            max_gamma = gamma
        print('Score C = ' + str(C) + '  gamma = ' + str(gamma) + '  score = ' + str(int(score*100)) +'%')

score = max_clf.score(X_test, y_test)
print('Score on test set: C = ' + str(max_C) + '  gamma = ' + str(max_gamma) + '  score = ' + str(int(score * 100)) + '%')
title = ('RBF kernel (C = ' + str(max_C) + '  gamma = ' + str(max_gamma) + ')')
print_graph(title, max_clf)

X_train = np.concatenate((X_train, X_val), axis=0)
y_train = np.concatenate((y_train, y_val), axis=0)

X_folds = np.array_split(X_train, 5)
y_folds = np.array_split(y_train, 5)

max_score = 0

print('Grid search of the best parameters for an RBF kernel using 5 folds:\n\n')

for k in range(5):

    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    for C in columns:
        row = []
        for gamma in rows:
            model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
            clf = model.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            if score >= max_score:
                max_score = score
                max_clf = clf
                max_C = C
                max_k = k
                max_gamma = gamma
            print('(' + str(k+1) + '-fold)' + 'Score C = ' + str(C) + '  gamma = ' + str(gamma) + '  score = ' + str(int(score * 100)) + '%')


title = ('RBF kernel (C = ' + str(max_C) + '  gamma = ' + str(max_gamma) + ' fold number = ' + str(max_k+1) + ')')
print_graph(title, max_clf)