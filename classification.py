from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

# Standardize the data
def standardize_data(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x_standardized = (x - mean) / std
    return x_standardized, mean, std

# Split the data into training 15%, tesing 15%, validation 70%
def split_data(x, y, state):
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.15, random_state=state)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.8235, random_state=state)  # 0.8235 * 0.85 ≈ 0.7
    return x_train, x_test, x_val, y_train, y_test, y_val

# Train a classifier based on a given model
def train_model(model_type, state, x_train, y_train):
    # Convert y_train to 1D array
    y_train_1d = y_train.values.ravel()
    
    if model_type == 'random_forest':
        clf = RandomForestClassifier(random_state=state)
    elif model_type == 'decision_tree':
        clf = DecisionTreeClassifier(random_state=state)
    elif model_type == 'knn':
        clf = KNeighborsClassifier()
    elif model_type == 'MLP':
        clf = MLPClassifier(random_state=state, max_iter=2000)
    elif model_type == 'SVM':
        clf = SVC(random_state=state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    clf.fit(x_train, y_train_1d)
    return clf

# Evaluate the model and return accuracy
def evaluate_model(clf, x, y):
    # Convert y to 1D array
    y_1d = y.values.ravel()
    return clf.score(x, y_1d)

# Calculate F1-score
def calculate_f1_score(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    y = y_test.values.ravel()
    return f1_score(y, y_pred, average='weighted')

# Calculate confusion matrix
def confusion_matrix(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    y = y_test.values.ravel()
    cm = sklearn_confusion_matrix(y, y_pred)
    return cm

# Print model details
def print_model_details(clf):
    print(clf)
    if hasattr(clf, 'feature_importances_'):
        print("Feature importances:", clf.feature_importances_)
    if hasattr(clf, 'n_layers_'):
        print("Number of layers in MLP:", clf.n_layers_)
    if hasattr(clf, 'support_'):
        print("Number of support vectors in SVM:", len(clf.support_))
    if hasattr(clf, 'n_neighbors'):
        print("Number of neighbors in KNN:", clf.n_neighbors)
    if hasattr(clf, 'n_estimators'):
        print("Number of trees in Random Forest:", clf.n_estimators)
    if hasattr(clf, 'max_depth'):
        print("Max depth of Decision Tree:", clf.max_depth)
    if hasattr(clf, 'C'):
        print("Regularization parameter C in SVM:", clf.C)
    if hasattr(clf, 'activation'):
        print("Activation function in MLP:", clf.activation)
