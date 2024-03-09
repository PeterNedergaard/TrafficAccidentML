import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier


def compare_algorithms():
    # Load dataset
    dataset = pd.read_csv('data/encoded_data.csv')
    X = dataset.drop('Accident_Severity', axis=1)
    y = dataset['Accident_Severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize TPOT
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)

    # Run the auto-ML process
    tpot.fit(X_train, y_train)

    # Evaluate on the test set
    print(tpot.score(X_test, y_test))


    # Split-out validation dataset
    # X = dataset.drop('Accident_Severity', axis=1)
    # Y = dataset['Accident_Severity']
    # # prepare configuration for cross validation test harness
    # seed = 7
    # # prepare models
    # models = []
    # models.append(('LR', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))
    # models.append(('RF', RandomForestClassifier()))
    # # evaluate each model in turn
    # results = []
    # names = []
    # scoring = 'accuracy'
    # for name, model in models:
    #     kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    #     cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)
    # # boxplot algorithm comparison
    # fig = plt.figure()
    # fig.suptitle('Algorithm Comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results)
    # ax.set_xticklabels(names)
    # plt.show()
