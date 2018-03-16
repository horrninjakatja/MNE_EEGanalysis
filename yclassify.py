#### Decoding Analysis over time ####

import numpy as np
from mne import decoding as dc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def classify(epochs):
    Xall = epochs.get_data()  # epochs*channels*time
    rel = epochs.events[:,2]!=2 # throw out standard
    X = Xall[rel,:,:] # classifying target vs. novelty
    y = epochs.events[rel, 2]  # target, standard or novelty

    clf = make_pipeline(StandardScaler(), LogisticRegression())
    clfest = dc.SlidingEstimator(clf, n_jobs=1, scoring='accuracy')

    values = dc.cross_val_multiscore(clfest, X, y, cv=5, n_jobs=1)
    return np.mean(values, axis=0)