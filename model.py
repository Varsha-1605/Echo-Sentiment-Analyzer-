from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pickle

from app import corpus, df


cv = CountVectorizer(max_features = 2500)


X = cv.fit_transform(corpus).toarray()
y  = df['feedback'].values

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Only Random Forest model
rf_model = RandomForestClassifier(n_estimators=50, random_state=2)

# Cross validation for Random Forest
scores = cross_validate(
    rf_model,
    X, y,
    cv=cv_strategy,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True
)

print("Random Forest with CountVectorizer:")
print("  Train Accuracy:", scores['train_accuracy'].mean())
print("  Test Accuracy :", scores['test_accuracy'].mean())
print("  Test Precision:", scores['test_precision'].mean())
print("  Test Recall   :", scores['test_recall'].mean())
print("  Test F1       :", scores['test_f1'].mean())


# Train Random Forest on full dataset
print("\nTraining Random Forest on full dataset...")
rf_model.fit(X, y)

# Save the trained Random Forest model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save the CountVectorizer (replace 'cv_vectorizer' with your actual variable name)
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

print("\nRandom Forest model saved as: random_forest_model.pkl")
print("CountVectorizer saved as: count_vectorizer.pkl")



