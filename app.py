# -*- coding: utf-8 -*-

import pandas as pd

# Mengunduh dan membaca dataset baru
data = pd.read_csv('Student_performance_data _.csv')

# Menampilkan beberapa baris pertama dari dataset
print(data.head())

# Menampilkan nama-nama kolom dalam dataset
print(data.columns)

import matplotlib.pyplot as plt
import seaborn as sns

# Mengatur gaya visualisasi
sns.set(style="whitegrid")

# Visualisasi distribusi fitur numerik
numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
for feature in numeric_features:
    plt.figure(figsize=(10, 5))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribusi {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frekuensi')
    plt.show()

# Visualisasi distribusi fitur kategorikal
categorical_features = data.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(y=data[feature])
    plt.title(f'Distribusi {feature}')
    plt.xlabel('Jumlah')
    plt.ylabel(feature)
    plt.show()

# Jika ada hubungan antara fitur yang ingin dianalisis
# Misalnya, jika kita ingin melihat hubungan antara dua fitur numerik
if len(numeric_features) >= 2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=numeric_features[0], y=numeric_features[1])
    plt.title(f'Relationship between {numeric_features[0]} and {numeric_features[1]}')
    plt.xlabel(numeric_features[0])
    plt.ylabel(numeric_features[1])
    plt.show()

"""Stage 2"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Gantilah 'GPA' dengan nama kolom target yang sesuai setelah Anda memeriksa nama kolom
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

"""Stage 3"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import numpy as np

# Model 1: Regresi Linier
linear_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', LinearRegression())])

# Model 2: Pohon Keputusan
tree_model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', DecisionTreeRegressor(random_state=42))])

# Model 3: K-Nearest Neighbors
knn_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', KNeighborsRegressor())])

# Melatih model dan evaluasi kinerja pada data uji
models = {'Linear Regression': linear_model,
          'Decision Tree': tree_model,
          'KNN': knn_model}

# Menyimpan hasil evaluasi
evaluation_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Menghitung metrik evaluasi
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Menghitung akurasi (dalam konteks klasifikasi)
    y_pred_classes = np.round(y_pred)  # Membulatkan prediksi ke kelas terdekat
    accuracy = accuracy_score(y_test, y_pred_classes)

    evaluation_results[name] = {
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae,
        'R-squared': r2,
        'Accuracy': accuracy
    }

    print(f"{name} Performance:")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")
    print(f"Accuracy: {accuracy}")
    print("\n")

# Menampilkan hasil evaluasi
evaluation_df = pd.DataFrame(evaluation_results).T
print(evaluation_df)

"""Visualisasi"""

import matplotlib.pyplot as plt
import numpy as np

# Menyiapkan data untuk plot
models = list(evaluation_results.keys())
mse = [evaluation_results[model]['Mean Squared Error'] for model in models]
rmse = [evaluation_results[model]['Root Mean Squared Error'] for model in models]
mae = [evaluation_results[model]['Mean Absolute Error'] for model in models]
r_squared = [evaluation_results[model]['R-squared'] for model in models]
accuracy = [evaluation_results[model]['Accuracy'] for model in models]

x = np.arange(len(models))

# Membuat bar plot
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

# Menghapus subplot yang kosong
fig.delaxes(axs[1, 2])

plt.tight_layout()
plt.show()

"""Penyetelan Hyperparameter"""

from sklearn.model_selection import GridSearchCV

# Definisikan parameter grid
param_grid = {
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Buat GridSearchCV untuk model Decision Tree
grid_search = GridSearchCV(tree_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Melatih GridSearchCV
grid_search.fit(X_train, y_train)

# Mendapatkan hyperparameter terbaik
best_params = grid_search.best_params_
print(f"Best hyperparameters for Decision Tree: {best_params}")

# Menggunakan model dengan hyperparameter terbaik
best_tree_model = grid_search.best_estimator_

# Evaluasi model dengan hyperparameter terbaik
y_pred = best_tree_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, np.round(y_pred))

print(f"Optimized Decision Tree Performance:")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
print(f"Accuracy: {accuracy}")

from sklearn.model_selection import cross_val_score, KFold

# Definisikan objek KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation untuk Regresi Linier
cv_scores_linear = cross_val_score(linear_model, X, y, cv=cv, scoring='neg_mean_squared_error')
print("\nCross-Validation Scores (Regresi Linier):", cv_scores_linear)
print("Mean CV Score (Regresi Linier):", -cv_scores_linear.mean())

# Cross-validation untuk Pohon Keputusan
cv_scores_tree = cross_val_score(tree_model, X, y, cv=cv, scoring='neg_mean_squared_error')
print("\nCross-Validation Scores (Pohon Keputusan):", cv_scores_tree)
print("Mean CV Score (Pohon Keputusan):", -cv_scores_tree.mean())

# Cross-validation untuk K-Nearest Neighbors
cv_scores_knn = cross_val_score(knn_model, X, y, cv=cv, scoring='neg_mean_squared_error')
print("\nCross-Validation Scores (K-Nearest Neighbors):", cv_scores_knn)
print("Mean CV Score (K-Nearest Neighbors):", -cv_scores_knn.mean())
