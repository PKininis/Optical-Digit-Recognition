from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import random
import numpy as np
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

def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    '''
    arr = X[index].reshape(16,16)
    plt.imshow(arr, cmap='gray')
    plt.show()
    return

def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''
    
    k = []
    arg = []
    
    for i in range(10):
        
     k = random.randint(0, len(y)-1) #Επιλέγω κάποιον τυχαίο ακέραιο ο οποίος θα χρησιμεύσει ως index για να βρω τυχαίους διαφορετικούς αριθμούς (labels) από το train set μου   
     
     while (y[k] != i): #έλεγχος για να πάρω όλα τα διαφορετικά ψηφία 
         
       k = random.randint(0, len(y)-1) 
         
     arg.append(X[k].reshape(16,16)) #παίρνω τα pixels του ψηφίου και τα βάζω σε μια λίστα αφού τα μετατρέψω σε κατάλληλους πίνακες 
     
     #plt.imshow(arg[i], cmap='gray')
     #plt.show()       
     
     fig = plt.figure()
          
    ax1 = fig.add_subplot(2, 5, 1)
    ax1.imshow(arg[0], cmap='gray')
    ax1.set_title('Digit 0')
    
    ax2 = fig.add_subplot(2, 5, 2)
    ax2.imshow(arg[1], cmap='gray')
    ax2.set_title('Digit 1')
    
    ax3 = fig.add_subplot(2, 5, 3)
    ax3.imshow(arg[2], cmap='gray')
    ax3.set_title('Digit 2')
    
    ax4 = fig.add_subplot(2, 5, 4)
    ax4.imshow(arg[3], cmap='gray')
    ax4.set_title('Digit 3')
    
    ax5 = fig.add_subplot(2, 5, 5)
    ax5.imshow(arg[4], cmap='gray')
    ax5.set_title('Digit 4')
    
    ax6 = fig.add_subplot(2, 5, 6)
    ax6.imshow(arg[5], cmap='gray')
    ax6.set_title('Digit 5')
    
    ax7 = fig.add_subplot(2, 5, 7)
    ax7.imshow(arg[6], cmap='gray')
    ax7.set_title('Digit 6')
    
    ax8 = fig.add_subplot(2, 5, 8)
    ax8.imshow(arg[7], cmap='gray')
    ax8.set_title('Digit 7')
    
    ax9 = fig.add_subplot(2, 5, 9)
    ax9.imshow(arg[8], cmap='gray')
    ax9.set_title('Digit 8')
    
    ax10 = fig.add_subplot(2, 5, 10)
    ax10.imshow(arg[9], cmap='gray')
    ax10.set_title('Digit 9')
    
    plt.show()
     
    return    
    
def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    '''
    counter = 0
    sum_of_pixels_of_number = 0
    for i in range(len(y)):
        if y[i] == digit:
            pixels_of_number = X[i].reshape(16,16)
            sum_of_pixels_of_number += pixels_of_number[pixel]
            counter += 1
    return sum_of_pixels_of_number/counter   
   
def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    counter = 0
    sum_of_pixels = 0
    sqr_sum = 0
    for i in range(len(y)):
        if y[i] == digit:
            
            pixels_of_number = X[i].reshape(16,16)
            sum_of_pixels += pixels_of_number[pixel]
            sqr_sum += (pixels_of_number[pixel])**2
            counter += 1
    return (sqr_sum/counter) - ((sum_of_pixels/counter)**2)  
           
def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    
    a = np.array(X[digit == y])
    mean_of_every_pixel = a.mean(axis = 0)
    return mean_of_every_pixel       
            
def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    a = np.array(X[digit == y])
    var_of_every_pixel = a.var(axis = 0)
    return var_of_every_pixel   

def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    dist = 0
    
    for i in range(len(s)):
        
        dist += (s[i]-m[i])**2
        
    return np.sqrt(dist)

def euclidean_distance_classifier(X, X_mean):
    '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
        
    prediction = np.empty(X.shape[0], dtype = int)    
    
    for i in range(X.shape[0]):
                
        temp = np.Inf
        
        for j in range(X_mean.shape[0]):
            
            dist = euclidean_distance(X[i,:], X_mean[j,:])
            
            if temp > dist:
            
              temp = dist
              prediction[i] = j
            
    return prediction
            
class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        mean_for_every_digit = np.empty([10, X.shape[1]])
        
        for i in range(10):
            
            mean_for_every_digit[i] = digit_mean(X, y, i)
            
        self.X_mean_ = mean_for_every_digit
            
        return self


    def predict(self, X):
        """
        
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        
        """
        self.prediction_ = euclidean_distance_classifier(X, self.X_mean_)
        return self.prediction_

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        
        """
        k = self.predict(X)
        diff = k - y
        counter = 0
        for i in range(y.shape[0]):  
          
          if diff[i] == 0:
            
            counter += 1
            
        return (counter/y.shape[0])*100 
            
def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    
    return scores

    
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    
    prob = []    
        
    for i in range(10):
        
        counter = 0
        
        for j in range(y.shape[0]):
        
           if y[j] == i:
            
             counter += 1
            
        prob.append(counter/y.shape[0])
        
    #unique, counts = np.unique(y, return_counts = True)        
    
    return (np.array((prob)))

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False, thermal=1e-9):
        
        self.use_unit_variance = use_unit_variance
        self.digits = np.arange(10)
        self.apriori = None
        self.char_means = None
        self.char_var = None
        self.thermal = thermal

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        apriori = calculate_priors(X, y)
        self.apriori = apriori
        
        mean_for_every_digit = np.empty([10, X.shape[1]])
        vars_for_every_digit = np.empty([10, X.shape[1]])
        
        for i in range(10):
            
            mean_for_every_digit[i] = digit_mean(X, y, i)
            
            if self.use_unit_variance == False:
                
              vars_for_every_digit[i] = digit_variance(X, y, i)
             
            else:
                
                vars_for_every_digit[i].fill(1.0)
                
        if self.use_unit_variance == False:
            
            temp = self.thermal*vars_for_every_digit.max()
            
        #it seems we have run into some kind of trouble with the NaN values of var
        if self.use_unit_variance == False:
            
          vars_for_every_digit += temp

        self.char_means = mean_for_every_digit
        self.char_var = vars_for_every_digit
             
        return self
        
    def log_gaussian(self, digit, j):
        
        mean = self.char_means[digit]
        var = self.char_var[digit]
        
        return -np.log(np.sqrt(2 * np.pi * var)) - (np.power(j - mean, 2) / (2 * var))
        
    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        prediction = []

        for j in X:
            
            temp = []
            
            for i in range(10):
                
                posterior = 0.0
                
                apriori = self.apriori[i]
                
                gauss_value = self.log_gaussian(i, j)
                    
                for k in gauss_value:
                
                 posterior += k
                
                posterior += np.log(apriori)
   
                temp.append(posterior) 

            prediction.append(self.digits[np.argmax(temp)])
        
        self.prediction = np.array(prediction)
                
        return self.prediction

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        prediction = self.predict(X)
        
        diff = prediction - y
        
        counter = 0
        
        for i in range(y.shape[0]):  
          
          if diff[i] == 0:
            
            counter += 1
            
        return (counter/y.shape[0])*100


class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, layers, n_features, n_digits, epochs, batch_sz, lrate):
        
        self.layers = layers
        self.n_features = n_features
        self.n_digits = n_digits
        self.epochs = epochs
        self.lrate = lrate
        self.batch_sz = batch_sz
        self.model = CustomNN(self.layers, self.n_features, self.n_digits)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lrate)
        self.validation = None

    def fit(self, X, y, split=0.0):
        
        if split == 0.0:
            
            nn_train = BlobData(X, y, trans=ToTensor())
            
        else:
           nn_X_train, nn_X_test, nn_y_train, nn_y_test = train_test_split(X, y, test_size=split)
        
           nn_train = BlobData(nn_X_train, nn_y_train, trans=ToTensor())
           nn_test = BlobData(nn_X_test, nn_y_test, trans=ToTensor())
        
           test_dl = DataLoader(nn_test, batch_size=self.batch_sz, shuffle=True)
           
        train_dl = DataLoader(nn_train, batch_size=self.batch_sz, shuffle=True)
        
        for epoch in range(self.epochs): # loop through dataset
          running_average_loss = 0
          for i, data in enumerate(train_dl): # loop thorugh batches
            X_batch, y_batch = data # get the features and labels
            self.optimizer.zero_grad() # ALWAYS USE THIS!! 
            out = self.model(X_batch) # forward pass
            loss = self.criterion(out, y_batch) # compute per batch loss 
            loss.backward() # compurte gradients based on the loss function
            self.optimizer.step() # update weights 
        
            running_average_loss += loss.detach().item()
        if split > 0.0:    
          self.model.eval() # turns off batchnorm/dropout ...
          acc = 0
          n_samples = 0
          
          with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
            for i, data in enumerate(test_dl):
             X_batch, y_batch = data # test data and labels
             out = self.model(X_batch) # get net's predictions
             val, y_pred = out.max(1) # argmax since output is a prob distribution
             acc += (y_batch == y_pred).sum().detach().item() # get accuracy
             n_samples += X_batch.size(0)
          
          self.validation = acc/n_samples
         
        return self
    
    def predict(self, X):

        test_X = torch.from_numpy(X).type(torch.FloatTensor)
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(test_X)
            val, y_pred = out.max(1)

        return y_pred.cpu().detach().numpy()

    def score(self, X, y):
        
        prediction = self.predict(X)
        diff = y - prediction
        counter = 0
        
        for i in range(y.shape[0]):  
          
          if diff[i] == 0:
            
            counter += 1
            
        return (counter/y.shape[0])*100
        
        
class ToTensor(object):
  """converts a numpy object to a torch tensor"""
  def __init__(self):
        pass
      
  def __call__(self, datum):
      x, y = datum[0], datum[1]
      
      newx = torch.from_numpy(x).type(torch.FloatTensor)
      newy = torch.from_numpy(np.asarray(y)).type(torch.LongTensor) # Otherwise an error occurs during training
      
      return newx, newy

class BlobData(Dataset):
    def __init__(self, X, y, trans=None):
        # all the available data are stored in a list
        self.data = list(zip(X, y))
        # we optionally may add a transformation on top of the given data
        # this is called augmentation in realistic setups
        self.trans = trans
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.trans is not None:
            
            return self.trans(self.data[idx])
        
        else:
            
            return self.data[idx]

class LinearWActivation(nn.Module):
    def __init__(self, in_features, out_features, activation='sigmoid'):
        super(LinearWActivation, self).__init__()
        self.f = nn.Linear(in_features, out_features)
        if activation == 'sigmoid':
            self.a = nn.Sigmoid()
        else:
            self.a = nn.ReLU()
            
    def forward(self, x):
        
        return self.a(self.f(x))
    

class CustomNN(nn.Module):
    
    def __init__(self, layers, n_features, n_classes, activation='sigmoid'):
        
        super(CustomNN, self).__init__()
        layers_in = [n_features] + layers 
        layers_out = layers + [n_classes]
        self.f = nn.Sequential(*[
            LinearWActivation(in_feats, out_feats, activation=activation)
            for in_feats, out_feats in zip(layers_in, layers_out)
        ])
        
        self.clf = nn.Linear(n_classes, n_classes)
                
    def forward(self, x):
        
        y = self.f(x)
        
        return self.clf(y)

        
def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf_svm_linear = SVC(kernel="linear")
    accuracy = evaluate_classifier(clf_svm_linear, X, y, folds=5)
    
    return accuracy

def evaluate_decision_tree_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf_decision_tree = DecisionTreeClassifier()
    accuracy = evaluate_classifier(clf_decision_tree, X, y, folds=5)
    
    return accuracy

def evaluate_logistic_regression_classifier(X, y, folds=5):
    
    clf_logistic_regression = LogisticRegression()
    accuracy = evaluate_classifier(clf_logistic_regression, X, y, folds=5)
    
    return accuracy    

def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf_svm_rbf = SVC(kernel="rbf")
    accuracy = evaluate_classifier(clf_svm_rbf, X, y, folds=5)
    
    return accuracy

def evaluate_random_forest_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf_random_forest = RandomForestClassifier(n_estimators=50)
    accuracy = evaluate_classifier(clf_random_forest, X, y, folds=5)
    
    return accuracy

def evaluate_triple_classifier(clf, X, y):
    
    accuracy = evaluate_classifier(clf, X, y, folds=5)
    
    return accuracy

def evaluate_knn_classifier(X, y, folds=5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf_knn  = KNeighborsClassifier(n_neighbors=1)
    accuracy = evaluate_classifier(clf_knn, X, y, folds=5)
    
    return accuracy
    

def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = GaussianNB()
    score = evaluate_classifier(clf, X, y, folds=5)
    
    return score
    
    
def evaluate_custom_nb_classifier(X, y, folds=5):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError
    
def evaluate_sigmoid_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = SVC(kernel="sigmoid")
    accuracy = evaluate_classifier(clf, X, y, folds)
    
    return accuracy    
    
def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    
    clf = EuclideanDistanceClassifier()
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    
    return scores
    
def evaluate_nn_classifier(X, y, layers, epochs, batch_sz, lrate, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = PytorchNNModel(layers, X.shape[1], 10, epochs, batch_sz, lrate)
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy   

def evaluate_voting_classifier(X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError
    
    

def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    
    bag_score = {}
    
    classifiers = {1 : SVC(kernel="linear"), 
               2 : SVC(kernel="rbf"),
               3 : SVC(kernel="sigmoid"), 
               4 : DecisionTreeClassifier(),
               5 : KNeighborsClassifier(n_neighbors=1),
               6 : EuclideanDistanceClassifier(),
               7 : GaussianNB(),
               8 : CustomNBClassifier(),
               9 : CustomNBClassifier(use_unit_variance=True),
               10 : LogisticRegression(),
               11 : RandomForestClassifier(n_estimators=50)}

    class_labels = {1 : 'Linear SVM',
               2 : 'RBF SVM',
               3 : 'Sigmoid SVM',
               4 : 'Decision Tree',
               5 : 'kNN (k = 1)',
               6 : 'Euclidean Distance',
               7 : 'Naive Bayes (sklearn)',
               8 : 'Naive Bayes (custom)',
               9 : 'Naive Bayes (unit variance)',
               10 : 'Logistic Regression',
               11 : 'Random forest'}

    for key in classifiers:
        
        clf = classifiers[key]
        bag_clf = BaggingClassifier(clf)
        score = evaluate_classifier(bag_clf, X, y, folds)
        bag_score[class_labels[key]] = np.mean(score)

    return bag_score
    
def plot_clf(clf, X, y):
    
    fig, ax = plt.subplots()
    
    # title for the plots
    
    # Set-up grid for plotting.
    
    X0, X1 = X[:, 0], X[:, 1]
    
    x_min, x_max = X0.min() - 1, X0.max() + 1
    
    y_min, y_max = X1.min() - 1, X1.max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    for i in range(10):
        dots = ax.scatter(X0[y == i], X1[y == i], label='Digit '+str(i), s=60, alpha=0.9, edgecolors='k')
    
    ax.set_ylabel('Feauture 1')
    ax.set_xlabel('Feauture 2')
    
    ax.set_xticks(())
    ax.set_yticks(())
    
    ax.set_title('Decision surface of Classifier')
    
    ax.legend()
    
    plt.show()
    
    return

def plot_confusion_matrices(X_tr, y_tr, X_te, y_te):
    
    classifiers = {1 : SVC(kernel="linear"), 
               2 : SVC(kernel="rbf"),
               3 : SVC(kernel="sigmoid"), 
               4 : DecisionTreeClassifier(),
               5 : KNeighborsClassifier(n_neighbors=1),
               6 : EuclideanDistanceClassifier(),
               7 : GaussianNB(),
               8 : CustomNBClassifier(),
               9 : CustomNBClassifier(use_unit_variance=True),
               10 : LogisticRegression(),
               11 : RandomForestClassifier(n_estimators=50)}

    class_labels = {1 : 'Linear SVM',
               2 : 'RBF SVM',
               3 : 'Sigmoid SVM',
               4 : 'Decision Tree',
               5 : 'kNN (k = 1)',
               6 : 'Euclidean Distance',
               7 : 'Naive Bayes (sklearn)',
               8 : 'Naive Bayes (custom)',
               9 : 'Naive Bayes (unit variance)',
               10 : 'Logistic Regression',
               11 : 'Random forest'}

    for key in classifiers:
        classifiers[key].fit(X_tr, y_tr)

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(40,30))

    digits = np.arange(10)

    for key, ax in zip(classifiers, axes.flatten()):
        
        plot_confusion_matrix(classifiers[key], X_te, y_te, ax=ax, display_labels=digits)
        ax.title.set_text(class_labels[key])
    
    fig.delaxes(axes[3,2])
    
    plt.tight_layout()  
    plt.show()
    return

def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0,1)):
    
    plt.figure()
    plt.title("Learning Curve")
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label="Cross-validation score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.legend(loc="best")
    plt.show()
    
    return plt