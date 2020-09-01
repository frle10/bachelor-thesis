'''
A script developed for easier classification of EEG signals for the bachelor's thesis.

@author Ivan Skorupan
'''

import sys
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

def fitModel(df, clf):
  y = df.diagnosis
  X = df.drop(['session_ID', 'diagnosis'], axis=1)

  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=97)

  yPred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
  scores = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'f1_weighted'], n_jobs=-1)
  accuracy = scores['test_accuracy'].mean()
  f1Weighted = scores['test_f1_weighted'].mean()

  labels = np.unique(y)
  confMat = confusion_matrix(y, yPred, labels=labels)
  confMatDf = pd.DataFrame(confMat, index=labels, columns=labels)
  return confMat, confMatDf, accuracy, f1Weighted

def printConfusionMatrixData(confMat, confMatDf):
  print("Confusion Matrix")
  print("-" * 30)
  print(confMatDf)

def truncateDiagnosesToFourClasses(df):
  truncatedDiagnosis = []
  for b in df.diagnosis:
    if b.startswith('F31'):
      truncatedDiagnosis.append('bipolarDisorder')
    elif b.startswith('F32'):
      truncatedDiagnosis.append('depressiveEpisodes')
    elif b.startswith('F33'):
      truncatedDiagnosis.append('recurrentDepressiveDisorder')
    elif b == 'healthy':
      truncatedDiagnosis.append('healthy')

  df.diagnosis = truncatedDiagnosis
  return df

# Main
def main():
  # Set some properties related to dataframe printing
  pd.set_option('display.max_colwidth', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.expand_frame_repr', False)

  if len(sys.argv) == 1:
    print("The script expects at least one argument (path to .csv file containing the dataset).")
    exit()

  csv = sys.argv[1]
  _, ext = os.path.splitext(csv)

  if ext != '.csv':
    print("Only files with extension .csv are supported.")
    exit()

  # Read the dataset
  print("Reading file '" + csv + "'.")
  print()
  try:
    df = pd.read_csv(csv)
  except:
    print("Could not read the .csv file.")
    exit()
  
  # Create an estimator
  if len(sys.argv) > 2:
    classifier = sys.argv[2]
    if classifier == 'rf':
      # clf = RandomForestClassifier(class_weight='balanced', max_depth=3)
      clf = RandomForestClassifier()
    elif classifier == 'svm':
      clf = SVC(class_weight='balanced')
      # clf = SVC(kernel='linear')
    elif classifier == 'knn':
      # clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
      clf = KNeighborsClassifier(n_neighbors=3)
    elif classifier == 'ada':
      clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm='SAMME')
  else:
    clf = RandomForestClassifier()
  
  # See if class truncation is necessary
  if len(sys.argv) > 3 and sys.argv[3] == 'notrunc':
    pass
  else:
    df = truncateDiagnosesToFourClasses(df)

  # Print class label distribution
  print(pd.value_counts(df.diagnosis))
  print()

  # Perform classification and print the results
  print("Performing multiclass classification using " + classifier + ".")
  print()
  confMat, confMatDf, accuracy, f1Weighted = fitModel(df, clf)
  printConfusionMatrixData(confMat, confMatDf)
  print()
  print("Accuracy: " + str(accuracy))
  print("F1 weighted: " + str(f1Weighted))

# Test script execution type
if __name__ == "__main__":
  main()
