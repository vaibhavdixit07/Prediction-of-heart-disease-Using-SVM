from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

dataset = genfromtxt('Actual Data With Empty Values.csv',delimiter=',')

X = dataset[:,0:12]
Y = dataset[:,13]
for index, item in enumerate(Y):     
	if not (item == 0.0):       
		Y[index] = 1
print(Y)
target_names = ['Non Diseased','Diseased']
def plot_2D(data,target,target_names):
	colors = cycle('rgbcmykw')
	target_ids = range(len(target_names))
	plt.figure()
	for i,c, label in zip(target_ids, colors, target_names):
		plt.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)
	plt.legend()
	plt.savefig('Problem 2 Graph')

pca = PCA(n_components=2, whiten=True).fit(X)
X_new = pca.transform(X)

plot_2D(X_new, Y, target_names)
plt.show()

modelSVM2 = SVC(C = 1.0,kernel='poly')
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(X_new, Y, test_size = 0.2, train_size=0.8, random_state=0)
modelSVM2 = modelSVM2.fit(X_train2,Y_train2)
accuracy2 = modelSVM2.score(X_test2,Y_test2)
print('Polynomial score with split = {0:.1f}%'.format(accuracy2*100))

modelSVM2 = SVC(C = 1.0,kernel='rbf')
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(X_new, Y, test_size = 0.2, train_size=0.8, random_state=0)
modelSVM2 = modelSVM2.fit(X_train2,Y_train2)
accuracy2 = modelSVM2.score(X_test2,Y_test2)
print('RBF score with split = {0:.1f}%'.format(accuracy2*100))

svc = SVC(C=1.0,kernel='rbf')
parameters = {'C': (100, 1e3, 1e4, 1e5),'gamma': (1e-08, 1e-7, 1e-6, 1e-5)}
grid_search = GridSearchCV(svc, parameters, n_jobs=-1, cv=5)
X_train6,X_test6,Y_train6,Y_test6 = train_test_split(X_new, Y, test_size = 0.2, train_size=0.8, random_state=0)
grid_search.fit(X_train6, Y_train6)
svc_best = grid_search.best_estimator_
accuracy = svc_best.score(X_test6, Y_test6)
print('Grid search:{0:.1f}%'.format(accuracy*100))

skf=StratifiedKFold(n_splits=6)
for train_index, test_index in skf.split(X_new, Y):
        X_train5,X_test5=X_new[train_index],X_new[test_index]
        Y_train5,Y_test5=Y[train_index],Y[test_index]

modelSVM5 = SVC(C=1.0, kernel='rbf')
modelSVM5 = modelSVM5.fit(X_train5,Y_train5)
accuracy5 = modelSVM5.score(X_test5,Y_test5)
print('Stratified K Fold Score = {0:.1f}%'.format(accuracy5*100))

prediction = modelSVM5.predict(X_test5)
report = classification_report(Y_test5, prediction)
print(report)

modelSVMRaw5 = SVC(C=1.0, kernel='rbf')
modelSVMRaw5 = modelSVMRaw5.fit(X_new,Y)
cnt2 = 0
for i in modelSVMRaw5.predict(X_new):
	if i == Y[1]:
		cnt2 = cnt2 + 1
accuracy_5 = float(cnt2)/270
print('On PCA valued X_new = {0:.1f}%'.format(accuracy_5*100))

X_min, X_max = X_new[:,0].min() - 1, X_new[:,0].max() + 1
Y_min, Y_max = X_new[:,1].min() - 1, X_new[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(X_min, X_max,0.2),np.arange(Y_min, Y_max,0.2))


plt .subplot(1,1, i + 1)
Z = modelSVM5.predict(np.c_[xx.ravel(), yy.ravel()])


Z = Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_new[:,0], X_new[:,1], c=Y,cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel("Features applied by PCA")
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xticks(())
plt.yticks(())
plt.legend()
titles = "Stratified K Fold SVC (RBF kernel)"
plt.title(titles)
plt.savefig('Graph')
plt.show()

train_sizes, train_scores, valid_scores = learning_curve(SVC(C=1.0, kernel='rbf'), X_new, Y, train_sizes=[20,40,60,80,100], cv=6)
train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
plt.xlabel("Training Set")
plt.ylabel("Score")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend()
plt.grid()
plt.title("Learning Curves (SVM, RBF kernel)")
plt.savefig('train')
plt.show()

StdSc= StandardScaler()
StdSc= StdSc.fit(X)
X_new= StdSc.transform(X)

modelSVM2 = SVC(C = 1.0,kernel='rbf')
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(X_new, Y, test_size = 0.2, train_size=0.8, random_state=0)
modelSVM2 = modelSVM2.fit(X_train2,Y_train2)
accuracy2 = modelSVM2.score(X_test2,Y_test2)
print('RBF score with split = {0:.1f}%'.format(accuracy2*100))