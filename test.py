import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from imblearn.over_sampling import SMOTE

# Read dataset
df = pd.read_csv('datasets/combinations_of_interest/FS_severe_dep&healthy.csv')

# truncatedDiagnosis = []
# for b in df.diagnosis:
#   if b.startswith('F31'):
#     truncatedDiagnosis.append('bipolarDisorder')
#   elif b.startswith('F33'):
#     truncatedDiagnosis.append('recurrentDepressiveDisorder')
#   elif b == 'F32.2' or b == 'F32.3':
#     truncatedDiagnosis.append('difficultDepressiveEpisodes')
#   elif b == 'healthy':
#     truncatedDiagnosis.append('healthy')
#   else:
#     truncatedDiagnosis.append('easyAndNormalDepression')

# df.diagnosis = truncatedDiagnosis

# Separate input features (X) and target variable (y)
y = df.label
X = df.drop(['session_ID', 'label'], axis=1)

print(pd.value_counts(y))
print()

# Apply SMOTE
# sm = SMOTE(random_state=42, k_neighbors=3)
# X, y = sm.fit_resample(X, y)

# Calculate class weights
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

# Create a cross-validation object
skf = StratifiedKFold()
print(skf.get_n_splits(X))

# Train model
clf = RandomForestClassifier(class_weight='balanced', max_depth=3)
scores = cross_validate(clf, X, y, cv=skf, scoring=['accuracy', 'f1_macro'])

print(scores['test_accuracy'])
print(scores['test_f1_macro'])
