import os
import cv2
import numpy as np
import pickle as pk

from feature import NPDFeature
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

DATA_DIR = './datasets/original'
FACE_DIR = os.path.join(DATA_DIR, 'face')
NONFACE_DIR = os.path.join(DATA_DIR, 'nonface')
DATA_PROCESSED_PATH = './npd.dat'

DATA_SIZE = 500
TRAIN_SAMPLES = 400
SCALE_SIZE = (24, 24)

def load_data():
    if os.path.exists(DATA_PROCESSED_PATH):
        with open(DATA_PROCESSED_PATH, 'rb') as f:
            return pk.load(f)

    imgs = []
    img_labels = []

    face_paths = os.listdir(FACE_DIR)
    for face_path in face_paths:
        img = cv2.imread(os.path.join(FACE_DIR, face_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, SCALE_SIZE, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        img_labels.append(+1)

    nonface_paths = os.listdir(NONFACE_DIR)
    for nonface_path in nonface_paths:
        img = cv2.imread(os.path.join(NONFACE_DIR, nonface_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, SCALE_SIZE, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        img_labels.append(-1)

    npd_features = []
    for img in imgs:
        npd = NPDFeature(img)
        npd_features.append(npd.extract())

    with open(DATA_PROCESSED_PATH, 'wb') as f:
        pk.dump((npd_features, img_labels), f)

    return npd_features, img_labels

def split_data(X, y):
    train_face_indices = np.random.choice(DATA_SIZE, TRAIN_SAMPLES, replace=False)
    train_nonface_indices = np.random.choice(DATA_SIZE, TRAIN_SAMPLES, replace=False) + 500
    train_indices = np.array(list(train_face_indices) + list(train_nonface_indices))
    test_indices = np.array(list(set(range(DATA_SIZE * 2)) - set(train_indices)))
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

if __name__ == "__main__":
    X, y = load_data()
    X_train, y_train, X_val, y_val = split_data(np.array(X), np.array(y))
    print(y_train.shape, y_val.shape)
    clf = AdaBoostClassifier(DecisionTreeClassifier, 10)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_val)
    print('Accuracy: ' + str((y_pre == y_val).astype(float).sum()/len(y_val)))
    print('F1 score: ' + str(f1_score(y_val, y_pre, labels=[1, -1], average='macro')))

