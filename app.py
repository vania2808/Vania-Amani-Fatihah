import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import numpy as np

# Setting visualization style
sns.set(style="whitegrid")

# Function to load data
@st.cache_data
def load_data():
    url = "Student_performance_data_.csv"
    return pd.read_csv(url)

# Loading data
data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Go to", ["Dataset Overview", "Data Visualization", "Model Training", "Hyperparameter Tuning"])

if pages == "Dataset Overview":
    st.title("Dataset Overview")
    st.write(data.head())
    st.write("Columns in dataset:", data.columns.tolist())

elif pages == "Data Visualization":
    st.title("Data Visualization")

    # Numeric Features Distribution
    st.header('Numerical Features Distribution')
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    for feature in numeric_features:
        plt.figure(figsize=(10, 5))
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        st.pyplot(plt)
        plt.close()

    # Categorical Features Distribution
    st.header('Categorical Features Distribution')
    categorical_features = data.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=data[feature])
        plt.title(f'Distribution of {feature}')
        plt.xlabel('Count')
        plt.ylabel(feature)
        st.pyplot(plt)
        plt.close()

elif pages == "Model Training":
    st.title("Model Training")

    # Splitting data into train and test sets
    st.write("Splitting the data into training and test sets...")
    X = data.drop('GPA', axis=1)
    y = data['GPA']
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing pipelines
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
    
    # Models
    models = {
        'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
        'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(random_state=42))]),
        'KNN': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', KNeighborsRegressor())])
    }
    
    # Training and evaluating models
    evaluation_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, np.round(y_pred))
        
        evaluation_results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Accuracy': accuracy
        }
    
    st.write("Evaluation Results:")
    st.write(pd.DataFrame(evaluation_results).T)
    
    # Plotting evaluation results
    st.write("Performance Metrics Comparison:")
    metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'Accuracy']
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for idx, metric in enumerate(metrics):
        ax = axs[idx // 3, idx % 3]
        values = [evaluation_results[model][metric] for model in models]
        ax.bar(models.keys(), values)
        ax.set_title(metric)
    
    # Removing empty subplot
    fig.delaxes(axs[1, 2])
    plt.tight_layout()
    st.pyplot(fig)

elif pages == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning")
    
    # Hyperparameter tuning for Decision Tree
    param_grid = {
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(models['Decision Tree'], param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    st.write("Best Parameters for Decision Tree:", grid_search.best_params_)
    
    best_tree_model = grid_search.best_estimator()
    y_pred = best_tree_model.predict(X_test)
    st.write("Performance of Tuned Decision Tree:")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred)}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    st.write(f"R2: {r2_score(y_test, y_pred)}")
    st.write(f"Accuracy: {accuracy_score(y_test, np.round(y_pred))}")

    # Cross-validation for all models
    st.write("Cross-validation results:")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        st.write(f"{name}: {cv_scores}")
        st.write(f"Mean CV Score: {-cv_scores.mean()}")

