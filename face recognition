# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:17:01 2018

@author: 吳嵩裕
"""

from sklearn.decomposition import PCA
import numpy as np
import os
import skimage.io as io
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from scipy.linalg import *

#eigenface
def plot_gallery(images, titles, h, w, n_row=10, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(count):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, i):

    return 'predicted: %s\ntrue:      %s' % (y_pred[i], y_test[i])







dim = 50
face = []
list1 = []
count = 10#輸出幾張eigen face(不大於dim)
h=112
w=92
n_classes = 40

#dirs=資料夾目錄
dirs = os.listdir('E:/Users/吳嵩裕/Desktop/碩班1上課程/圖形識別/人臉作業/att_faces')
#讀取目錄裡每個子資料夾中的10張圖拿來當資料
for folders in dirs: 
    mypath = 'E:/Users/吳嵩裕/Desktop/碩班1上課程/圖形識別/人臉作業/att_faces'
    mypath = mypath + '/' + folders + '/*.pgm'
    coll = io.ImageCollection(mypath)#圖片的collection集合
    for num in range(10):
        im = np.reshape(coll[num], (1, 10304))
        face.append(im)
#排列成400個樣本，然後配上40個id
face_id = []
k=0
face = np.reshape(face, (400,10304))
face = np.array(face.tolist())
for i in range(40):
    for j in range(10):
        face_id.insert(k,i)
#分成50%訓練資料，50%辨識資料
X_train, X_test, y_train, y_test = train_test_split(face, face_id, 
                                                    test_size=0.5, random_state=42)


#做PCA
pca = PCA(n_components=dim, svd_solver='arpack',whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#做SVM#預測
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
confusion = confusion_matrix(y_test, y_pred, labels=range(n_classes))
#預測與比較圖和基準臉
eigenfaces = pca.components_.reshape((dim, h, w))
prediction_titles = [title(y_pred, y_test, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

#PCA後的貝氏預測
gnb = GaussianNB()
gnb.fit(X_train_pca, y_train)
o1 = gnb.predict(X_test_pca)
sum0 = 0;
for i in range(len(y_test)):
    if(o1[i]==y_test[i]):sum0+=1
print(confusion_matrix(y_test, o1, labels=range(n_classes)))
confusion1 = confusion_matrix(y_test, o1, labels=range(n_classes))
print('PCA.gaussian accuracy:',sum0/200)

#經PCA後的樣本做FLDA，然後貝氏預測
sklearn_lda = LDA(n_components=dim)
X_lda_sklearn = sklearn_lda.fit(X_train_pca, y_train)
o = sklearn_lda.predict(X_test_pca)
sum1 = 0
for i in range(len(y_test)):
    if(o[i]==y_test[i]):sum1+=1
print(confusion_matrix(y_test, o, labels=range(n_classes)))
confusion2 = confusion_matrix(y_test, o, labels=range(n_classes))
print('LDA.gaussian accuracy:',sum1/200)




  
  
      
