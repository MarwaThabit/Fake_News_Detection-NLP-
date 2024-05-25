from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
from sklearn.metrics import accuracy_score

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def random_forest_classifier(X_train, y_train):
    model = RandomForestClassifier(n_estimators=120, max_depth=None,random_state=1)
    model.fit(X_train, y_train)
    return model

C=2
def svm_classifier(X_train, y_train):
    model = SVC(C=C )
    model.fit(X_train, y_train)
    return model

def knn_classifier(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=7, weights='distance')
    model.fit(X_train, y_train)
    return model


def svm_linear_classifier(X_train, y_train):
    model = SVC(kernel='linear',C=C)
    model.fit(X_train, y_train)
    return model

def svm_polynomial_classifier(X_train, y_train, degree=3):
    model = SVC(kernel='poly', degree=degree,C=C)
    model.fit(X_train, y_train)
    return model

def svm_rbf_classifier(X_train, y_train):
    model = SVC(kernel='rbf', gamma='scale',C=2)
    model.fit(X_train, y_train)
    return model


def decision_tree_classifier(X_train, y_train):
    model = DecisionTreeClassifier(random_state=1,max_depth=3)
    model.fit(X_train, y_train)
    return model

def logistic_regression_classifier(X_train, y_train):
    model = LogisticRegression(random_state=1,C=2)
    model.fit(X_train, y_train)
    return model


def passive_model(tfidf_train,y_train):
    pac = PassiveAggressiveClassifier(C=3,max_iter=20,random_state=3)
    pac.fit(tfidf_train, y_train)
    save_model(pac,'PassiveAggressiveClassifier.pkl')
    y_train_pred = pac.predict(tfidf_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    return train_accuracy

