import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
import streamlit as st

# Mengunduh dan membaca dataset
@st.cache_data
def load_data():
    return pd.read_csv('Student_performance_data_.csv')

data = load_data()

# Menampilkan beberapa baris pertama dari dataset
st.write("### Dataset Preview")
st.write(data.head())

# Menampilkan nama-nama kolom dalam dataset
st.write("### Column Names")
st.write(data.columns)

# Mengatur gaya visualisasi
sns.set(style="whitegrid")

# Visualisasi distribusi fitur numerik
st.write("### Distribusi Fitur Numerik")
numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
for feature in numeric_features:
    st.subheader(f'Distribusi {feature}')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data[feature], kde=True, ax=ax)
    plt.title(f'Distribusi {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frekuensi')
    st.pyplot(fig)

# Visualisasi distribusi fitur kategorikal
st.write("### Distribusi Fitur Kategorikal")
categorical_features = data.select_dtypes(include=['object']).columns
for feature in categorical_features:
    st.subheader(f'Distribusi {feature}')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(y=data[feature], ax=ax)
    plt.title(f'Distribusi {feature}')
    plt.xlabel('Jumlah')
    plt.ylabel(feature)
    st.pyplot(fig)

# Jika ada hubungan antara fitur yang ingin dianalisis
# Misalnya, jika kita ingin melihat hubungan antara dua fitur numerik
if len(numeric_features) >= 2:
    st.write("### Hubungan Antara Dua Fitur Numerik")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=numeric_features[0], y=numeric_features[1], ax=ax)
    plt.title(f'Relationship between {numeric_features[0]} and {numeric_features[1]}')
    plt.xlabel(numeric_features[0])
    plt.ylabel(numeric_features[1])
    st.pyplot(fig)

# Stage 2: Data Preparation
X = data.drop('GPA', axis=1)  # Ganti 'GPA' dengan nama kolom target Anda
y = data['GPA']  # Ganti 'GPA' dengan nama kolom target Anda

# Mengubah nilai kategori dalam kolom target menjadi nilai numerik (jika diperlukan)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menentukan praproses untuk fitur numerik dan kategorikal
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

# Stage 3: Model Training and Evaluation
linear_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', LinearRegression())])

tree_model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', DecisionTreeRegressor(random_state=42))])

knn_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', KNeighborsRegressor())])

models = {'Linear Regression': linear_model,
          'Decision Tree': tree_model,
          'KNN': knn_model}

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

# Menampilkan hasil evaluasi
st.write("### Evaluation Results")
evaluation_df = pd.DataFrame(evaluation_results).T
st.write(evaluation_df)

# Visualisasi Evaluasi
st.write("### Model Performance Visualization")
mse = [evaluation_results[model]['Mean Squared Error'] for model in models]
rmse = [evaluation_results[model]['Root Mean Squared Error'] for model in models]
mae = [evaluation_results[model]['Mean Absolute Error'] for model in models]
r_squared = [evaluation_results[model]['R-squared'] for model in models]
accuracy = [evaluation_results[model]['Accuracy'] for model in models]

x = np.arange(len(models))

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].bar(x, mse, color='b', width=0.4)
axs[0, 0].set_title('Mean Squared Error')
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(models)

axs[0, 1].bar(x, rmse, color='g', width=0.4)
axs[0, 1].set_title('Root Mean Squared Error')
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(models)

axs[0, 2].bar(x, mae, color='r', width=0.4)
axs[0, 2].set_title('Mean Absolute Error')
axs[0, 2].set_xticks(x)
axs[0, 2].set_xticklabels(models)

axs[1, 0].bar(x, r_squared, color='c', width=0.4)
axs[1, 0].set_title('R-squared')
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(models)

axs[1, 1].bar(x, accuracy, color='m', width=0.4)
axs[1, 1].set_title('Accuracy')
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(models)

fig.delaxes(axs[1, 2])

plt.tight_layout()
st.pyplot(fig)

# Penyetelan Hyperparameter
param_grid = {
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(tree_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
st.write(f"### Best hyperparameters for Decision Tree: {best_params}")

best_tree_model = grid_search.best_estimator_

y_pred = best_tree_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, np.round(y_pred))

st.write(f"### Optimized Decision Tree Performance:")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Root Mean Squared Error: {rmse}")
st.write(f"Mean Absolute Error: {mae}")
st.write(f"R-squared: {r2}")
st.write(f"Accuracy: {accuracy}")

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_linear = cross_val_score(linear_model, X, y, cv=cv, scoring='neg_mean_squared_error')
st.write("\n### Cross-Validation Scores (Regresi Linier):", cv_scores_linear)
st.write("Mean CV Score (Regresi Linier):", -cv_scores_linear.mean())

cv_scores_tree = cross_val_score(tree_model, X, y, cv=cv, scoring='neg_mean_squared_error')
st.write("\n### Cross-Validation Scores (Pohon Keputusan):", cv_scores_tree)
st.write("Mean CV Score (Pohon Keputusan):", -cv_scores_tree.mean())

cv_scores_knn = cross_val_score(knn_model, X, y, cv=cv, scoring='neg_mean_squared_error')
st.write("\n### Cross-Validation Scores (K-Nearest Neighbors):", cv_scores_knn)
st.write("Mean CV Score (K-Nearest Neighbors):", -cv_scores_knn.mean())
