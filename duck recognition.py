# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:51:53 2017

@author: 吳嵩裕
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
import cv2
import numpy.random as npr 
d=[1]*2
l=[1]*2
img = cv2.imread("d.jpg", -1) #-1是讀彩色
print(img)
#cv2.imshow("GG", img)
#cv2.waitKey(0)
gnb = GaussianNB()
r,c,dim = np.array(img).shape
#print(d)
duck_list = []
#print(d,r,c)
for i in range(r):
    for j in range(c):
        #if ((img[i,j,0]!=255) and (img[i,j,1]!=255) and (img[i,j,2]!=255)):
            duck_list.append(img[i][j])
#print(duck_list)

mean_v = np.mean(duck_list, axis=0) #axis=0 對橫的做
#print(mean_v)
t =np.array(duck_list).T
covmat = np.cov(t)
#print(covmat)

d[0] = npr.multivariate_normal(mean_v, covmat, 100)
l[0] = np.ones(100)*1
#print(d[0])
#print("mean vec = ", mean_v)
#print("cov mat = \n", covmat)

img = cv2.imread("nd.jpg", -1) #-1是讀彩色
gnb = GaussianNB()
r,c,dim = np.array(img).shape
#print(d)
nonduck_list = []
#print(d,r,c)
for i in range(r):
    for j in range(c):
        if ((img[i,j,0]!=255) and (img[i,j,1]!=255) and (img[i,j,2]!=255)):
            nonduck_list.append(img[i][j])
#print(nonduck_list)

nmean_v = np.mean(nonduck_list, axis=0) #axis=0 對橫的做
#print(nmean_v)
t =np.array(nonduck_list).T
ncovmat = np.cov(t)
#print(ncovmat)
d[1] = npr.multivariate_normal(nmean_v, ncovmat, 100)
l[1] = np.ones(100)*2
#print(d[1].shape)
#print("mean vec = ", nmean_v)
#print("cov mat = \n", ncovmat)

data = np.concatenate(d)
#print(data)
labels = np.concatenate(l)
#print(labels)
gnb.fit(data, labels)
img = cv2.imread("full_duck.jpg", -1)
r,c,dim = img.shape
new_img = []
for i in range(r):
    for j in range(c):
        new_img.append(img[i][j])
pred = gnb.predict(new_img)

for i in range(r*c):
    if pred[i]==2:
        a = i//c
        b = i%c
        img[a][b][0] = 0
        img[a][b][1] = 0
        img[a][b][2] = 0

#cv2.imshow("GG", img)
cv2.imwrite("D:\\GG.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
