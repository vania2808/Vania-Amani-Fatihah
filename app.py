import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV

# Set style for plots
sns.set(style="whitegrid")

# Load and display the dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\UAS_MPML\restaurant_menu_optimization_data.csv')
    return data

data = load_data()
st.write("Dataset Preview:")
st.write(data.head())

# Data preprocessing
X = data.drop('Profitability', axis=1)
y = data['Profitability']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(random_state=42))]),
    'K-Nearest Neighbors': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', KNeighborsRegressor())])
}

# Training and evaluation
evaluation_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    y_pred_classes = np.round(y_pred)
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    evaluation_results[name] = {
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae,
        'R-squared': r2,
        'Accuracy': accuracy
    }

# Display evaluation results
st.write("Model Evaluation Results:")
evaluation_df = pd.DataFrame(evaluation_results).T
st.write(evaluation_df)

# Hyperparameter tuning for Decision Tree
st.write("Tuning Hyperparameters for Decision Tree...")
param_grid = {
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(models['Decision Tree'], param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
st.write(f"Best hyperparameters for Decision Tree: {best_params}")

best_tree_model = grid_search.best_estimator_

y_pred = best_tree_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, np.round(y_pred))

st.write("Optimized Decision Tree Performance:")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Root Mean Squared Error: {rmse}")
st.write(f"Mean Absolute Error: {mae}")
st.write(f"R-squared: {r2}")
st.write(f"Accuracy: {accuracy}")

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, model in models.items():
    cv_scores[name] = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')

st.write("Cross-Validation Scores:")
for name, scores in cv_scores.items():
    st.write(f"{name}: Mean CV Score = {-scores.mean()}")
