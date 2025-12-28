##Pattern recognition, 1st Project
import pat_rec_functions as lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

traindf = pd.read_csv('train.txt', sep=' ', header=None) 
testdf = pd.read_csv('test.txt', sep=' ', header=None) 

traindf.drop(traindf.columns[257], axis = 1, inplace = True) 
testdf.drop(testdf.columns[257], axis = 1, inplace = True) 

y_test = testdf.iloc[:,0].to_numpy() 
y_train = traindf.iloc[:,0].to_numpy()
X_train = traindf.iloc[:,1:].values
X_test = testdf.iloc[:,1:].values

y_test = y_test.astype(int) 
y_train = y_train.astype(int)

print('Step 1 is completed, all data have been loaded.')

lib.show_sample(X_train, 131)

print('Step 2 is completed.')

a = lib.plot_digits_samples(X_train, y_train)

print('Step 3 is completed.')

# mean for every feauture of class 0

b = lib.digit_mean_at_pixel(X_train, y_train, 0, pixel=(10, 10))
print('the mean of the pixel [10][10] for the digit 0 is equal to:', b)

print('Step 4 is completed.')

#variance for every feauture of class 0

c = lib.digit_variance_at_pixel(X_train, y_train, 0, pixel=(10, 10)) 
print('the variance of the pixel [10][10] for the digit 0 is equal to:', c)

print('Step 5 is completed.')

#mean of every pixel

d = lib.digit_mean(X_train, y_train, 0)

#variance of every pixel

e = lib.digit_variance(X_train, y_train, 0)

print('Step 6 is completed.')

plt.imshow(d.reshape(16,16), cmap='gray')
plt.show()

print('Step 7 is completed.')

plt.imshow(e.reshape(16,16), cmap='gray')
plt.show()

print('Step 8 is completed.')

#(α) κατασκευάζω έναν πίνακα με στήλες τους μέσους όρους/vars των χαρακτηριστικών έκαστου ψηφίου. Κάθε γραμμή αντιστοιχεί σε ένα ψηφίο (10 γραμμές)

mean_for_every_digit = np.empty([10, X_train.shape[1]])
vars_for_every_digit = np.empty([10, X_train.shape[1]])

for i in range(10):
    
    mean_for_every_digit[i] = lib.digit_mean(X_train, y_train, i)
    vars_for_every_digit[i] = lib.digit_variance(X_train, y_train, i)

fig = plt.figure()
          
ax1 = fig.add_subplot(2, 5, 1)
ax1.imshow(mean_for_every_digit[0,:].reshape(16,16), cmap='gray')
ax1.set_title('Digit 0')
    
ax2 = fig.add_subplot(2, 5, 2)
ax2.imshow(mean_for_every_digit[1,:].reshape(16,16), cmap='gray')
ax2.set_title('Digit 1')
    
ax3 = fig.add_subplot(2, 5, 3)
ax3.imshow(mean_for_every_digit[2,:].reshape(16,16), cmap='gray')
ax3.set_title('Digit 2')
    
ax4 = fig.add_subplot(2, 5, 4)
ax4.imshow(mean_for_every_digit[3,:].reshape(16,16), cmap='gray')
ax4.set_title('Digit 3')
    
ax5 = fig.add_subplot(2, 5, 5)
ax5.imshow(mean_for_every_digit[4,:].reshape(16,16), cmap='gray')
ax5.set_title('Digit 4')
    
ax6 = fig.add_subplot(2, 5, 6)
ax6.imshow(mean_for_every_digit[5,:].reshape(16,16), cmap='gray')
ax6.set_title('Digit 5')
    
ax7 = fig.add_subplot(2, 5, 7)
ax7.imshow(mean_for_every_digit[6,:].reshape(16,16), cmap='gray')
ax7.set_title('Digit 6')
    
ax8 = fig.add_subplot(2, 5, 8)
ax8.imshow(mean_for_every_digit[7,:].reshape(16,16), cmap='gray')
ax8.set_title('Digit 7')
    
ax9 = fig.add_subplot(2, 5, 9)
ax9.imshow(mean_for_every_digit[8,:].reshape(16,16), cmap='gray')
ax9.set_title('Digit 8')
    
ax10 = fig.add_subplot(2, 5, 10)
ax10.imshow(mean_for_every_digit[9,:].reshape(16,16), cmap='gray')
ax10.set_title('Digit 9')
    
plt.show()
    
print('Step 9 is completed.')

#number 9
f = lib.euclidean_distance_classifier(X_test, mean_for_every_digit)

print('The number 101 digit has been classified as {} but its actual value is {}.'.format(f[101],y_test[101]))

print('Step 10 is completed.')

#Prdeiction for x_test

k = lib.euclidean_distance_classifier(X_test, mean_for_every_digit)

#Calculation of our success rate

counter = 0
diff = []

for i in range(y_test.shape[0]):
    
  y = np.empty(y_test.shape[0], dtype = int)
  y = k - y_test
  
  if y[i] == 0:
      
      counter += 1
      
success_rate = (counter/y_test.shape[0])*100

print('the success rate of the Euclidian classifier is equal to: ', success_rate, '%')

print('Step 11 is completed.')

lamda = lib.EuclideanDistanceClassifier()
lamda.fit(X_train, y_train)
score = lamda.score(X_test, y_test)
print(score)

print('Step 12 is completed.')

#5 fold CV of the Euclidian classifier

m = lib.evaluate_euclidean_classifier(X_train, y_train, folds=5)
print(m)
opa =  np.mean(m)
print(opa)

#12c
#learning_curve

classifier = lib.EuclideanDistanceClassifier()
classifier.fit(X_train, y_train)

train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=5, n_jobs=-1, train_sizes = np.linspace(.1, 1.0, 5))

lib.plot_learning_curve(train_scores/100, test_scores/100, train_sizes, ylim=(.8, .9))

#I use PCA in order to reduce the dimensions of the features in order to be able to draw the decision surface.

pca = PCA(n_components=2)

X_train_2 = pca.fit_transform(X_train)
X_test_2 = pca.fit_transform(X_test)

#I have to retrain my new model through my new data

model = lib.EuclideanDistanceClassifier()
model.fit(X_train_2, y_train)
valid = model.score(X_test_2, y_test)
print(valid)
lib.plot_clf(model, X_test_2, y_test)

print('Step 13 is completed.')

n = lib.calculate_priors(X_train, y_train)

o = np.mean(n)
p = np.std(n)

print(o, p)

print('Step 14 is completed.')

#Custom Naive Bayes Classifier

naive = lib.CustomNBClassifier()
naive.fit(X_train,y_train)
naive_score = naive.score(X_test,y_test)
print('The score of the Naive Bayes classifier is equal to: ', naive_score, '%')

#learning curve for Custom Naive Bayes Classifier

train_sizes_2, train_scores_2, test_scores_2 = learning_curve(naive, X_train, y_train, cv=5, n_jobs=-1, train_sizes = np.linspace(.1, 1.0, 5))

lib.plot_learning_curve(train_scores_2/100, test_scores_2/100, train_sizes_2, ylim=(.3, .99))

#Naive bayes taken from sklearn

clf_NBS = GaussianNB()
clf_NBS.fit(X_train, y_train)
print(clf_NBS.score(X_test, y_test))

#5 fold CV for Naive bayes taken from sklearn

score_for_sks_GNB = lib.evaluate_sklearn_nb_classifier(X_test, y_test, folds=5)
print('The score of the Naive Bayes classifier from SKlearn is equal to: ', np.mean(score_for_sks_GNB)*100, '%')
   
print('Step 15 is completed.')

##Custom Naive Bayes classifier with variance equal to 1

clf_NB1 = lib.CustomNBClassifier(use_unit_variance=True)
clf_NB1.fit(X_train,y_train)
naive_score_1 = clf_NB1.score(X_test,y_test)

print('The score of the Naive Bayes classifier is equal to: ', naive_score_1, '%')

print('Step 16 is completed.')

#Calculation of accuracy for various classifiers

clf_knn = KNeighborsClassifier(n_neighbors=1)
clf_knn.fit(X_train,y_train)
knn_score_0 = clf_knn.score(X_test, y_test)
knn_score_1 = lib.evaluate_knn_classifier(X_train, y_train, 5)

print(np.mean(knn_score_1))

clf_svm_rbf = SVC(kernel="rbf")
clf_svm_rbf.fit(X_train,y_train)
svm_rbf_score_0 = clf_knn.score(X_test, y_test)
svm_rbf_score_1 = lib.evaluate_rbf_svm_classifier(X_train, y_train, 5)

print(np.mean(svm_rbf_score_1))

clf_svm_linear = SVC(kernel="linear")
clf_svm_linear.fit(X_train,y_train)
svm_linear_score_0 = clf_svm_linear.score(X_test, y_test)
svm_linear_score_1 = lib.evaluate_linear_svm_classifier(X_train, y_train, 5)

print(np.mean(svm_linear_score_1))

clf_svm_sigmoid = SVC(kernel="sigmoid")
clf_svm_sigmoid.fit(X_train,y_train)
svm_sigmoid_score_0 = clf_svm_sigmoid.score(X_test, y_test)
svm_sigmoid_score_1 = lib.evaluate_sigmoid_svm_classifier(X_train, y_train, 5)

print(np.mean(svm_sigmoid_score_1))

clf_decision_tree = DecisionTreeClassifier()
clf_decision_tree.fit(X_train,y_train)
decision_tree_score_0 = clf_decision_tree.score(X_test, y_test)
decision_tree_score_1 = lib.evaluate_decision_tree_classifier(X_train, y_train, 5)

print(np.mean(decision_tree_score_1))

clf_random_forest = RandomForestClassifier(n_estimators=50)
clf_random_forest.fit(X_train,y_train)
random_forest_score_0 = clf_random_forest.score(X_test, y_test)
random_forest_score_1 = lib.evaluate_random_forest_classifier(X_train, y_train, 5)

print(np.mean(random_forest_score_1))

clf_logistic_reg = LogisticRegression()
clf_logistic_reg.fit(X_train,y_train)
logistic_reg_score_0 = clf_logistic_reg.score(X_test, y_test)
logistic_reg_score_1 = lib.evaluate_logistic_regression_classifier(X_train, y_train, 5)

print(np.mean(logistic_reg_score_1))

print('Step 17 is completed.')

#confusion matrix, shows the number of samples that get missclassified in each class.

lib.plot_confusion_matrices(X_train,y_train,X_test,y_test)

#hard voting for KNN, linear and rbf.

triple_clf_hard = VotingClassifier(estimators = [('linear', SVC(probability=True, kernel="linear")), ('rbf', SVC(probability=True, kernel="rbf")), ('knn', KNeighborsClassifier(n_neighbors=1))], voting = 'hard')
triple_clf_hard_score = lib.evaluate_classifier(triple_clf_hard, X_train, y_train, folds=5)

print(np.mean(triple_clf_hard_score))

#soft voting for KNN, linear and rbf.

triple_clf_soft = VotingClassifier(estimators = [('linear', SVC(probability=True, kernel="linear")), ('rbf', SVC(probability=True, kernel="rbf")), ('knn', KNeighborsClassifier(n_neighbors=1))], voting = 'soft')
triple_clf_soft_score = lib.evaluate_classifier(triple_clf_soft, X_train, y_train, folds=5)

print(np.mean(triple_clf_soft_score))

#Bagging_Classifier

zeta = lib.evaluate_bagging_classifier(X_train, y_train, folds=5)
print(zeta)

print('Step 18 is completed.')

##trials for different epochs, learning rates and batches

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 200, 32, 1e-2)
nnclf.fit(X_train, y_train, split=0.2)
nn1 = nnclf.validation*100

print('Setting the number of epochs equal to 200 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 300, 32, 1e-2)
nnclf.fit(X_train, y_train, split=0.2)
nn2 = nnclf.validation*100

print('Setting the number of epochs equal to 300 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 400, 32, 1e-2)
nnclf.fit(X_train, y_train, split=0.2)
nn3 = nnclf.validation*100

print('Setting the number of epochs equal to 400 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 500, 32, 1e-2)
nnclf.fit(X_train, y_train, split=0.2)
nn4 = nnclf.validation*100

print('Setting the number of epochs equal to 500 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 300, 16, 1e-2)
nnclf.fit(X_train, y_train, split=0.2)
nn5 = nnclf.validation*100

print('Setting the number of batches equal to 16 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 300, 32, 1e-2)
nnclf.fit(X_train, y_train, split=0.2)
nn6 = nnclf.validation*100

print('Setting the number of batches equal to 32 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 300, 64, 1e-2)
nnclf.fit(X_train, y_train, split=0.2)
nn7 = nnclf.validation*100

print('Setting the number of batches equal to 64 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 300, 128, 1e-2)
nnclf.fit(X_train, y_train, split=0.2)
nn8 = nnclf.validation*100

print('Setting the number of batches equal to 128 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 300, 32, 1e-3)
nnclf.fit(X_train, y_train, split=0.2)
nn9 = nnclf.validation*100

print('Setting the number of training rate equal to 0.001 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 300, 32, 1e-2)
nnclf.fit(X_train, y_train, split=0.2)
nn10 = nnclf.validation*100

print('Setting the number of training rate equal to 0.01 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

nnclf = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 300, 32, 1e-1)
nnclf.fit(X_train, y_train, split=0.2)
nn11 = nnclf.validation*100

print('Setting the number of training rate equal to 0.1 leads to a validation score of {}%.'.format(100*nnclf.validation))
print('This model has an accuracy of {}% for the actual test data.'.format(nnclf.score(X_test, y_test)))

epochs = [200, 300, 400, 500]
y_epochs = [nn1, nn2, nn3, nn4]

batches = [16, 32, 64, 128]
y_batches = [nn5, nn6, nn7, nn8]

learning_rate = [0.001, 0.01, 0.1]
y_learning_rate = [nn9, nn10, nn11]

fig, axs = plt.subplots(1, 3)
axs[0].plot(epochs, y_epochs)
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Score for different epochs')
axs[1].plot(batches, y_batches)
axs[1].set_xlabel('Batches')
axs[1].set_ylabel('Score for different batches')
axs[2].plot(learning_rate, y_learning_rate)
axs[2].set_xlabel('learning_rate')
axs[2].set_ylabel('Score for different learning rates')

plt.show()

##5-fold CV 

nnscore = lib.evaluate_nn_classifier(X_train, y_train, [100,100], 400, 32, 1e-2, folds=5)
keep = np.mean(nnscore)

clfrealdata = lib.PytorchNNModel([100,100], X_train.shape[1], 10, 400, 32, 1e-2)
clfrealdata.fit(X_train, y_train, split=0.0)
print('The accuracy of this model on the actual test data is {}%.'.format(clfrealdata.score(X_test,y_test)))

#learning_curve για την συνάρτηση που εκπαιδεύσαμε στα δεδομένα μας

train_sizes, train_scores, test_scores = learning_curve(clfrealdata, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
lib.plot_learning_curve(train_scores/100, test_scores/100, train_sizes, ylim=(0.0, 1.0))


print('Step 19 is completed.')











