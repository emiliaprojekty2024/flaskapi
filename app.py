from flask import Flask, jsonify, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import io

app = Flask(__name__)

# Wczytanie danych
data = pd.read_csv('meteo.csv')

# Filtracja tylko kolumn numerycznych do analizy
numeric_columns = ['temperatura', 'predkosc_wiatru', 'kierunek_wiatru', 'wilgotnosc_wzgledna', 'suma_opadu', 'cisnienie']
data_numeric = data[numeric_columns].dropna()

# Statystyki opisowe
descriptive_stats = data_numeric.describe()

# Macierz korelacji
correlation_matrix = data_numeric.corr()

# Model regresji liniowej
X_linear = data_numeric[['predkosc_wiatru', 'kierunek_wiatru', 'wilgotnosc_wzgledna', 'suma_opadu', 'cisnienie']]
y_linear = data_numeric['temperatura']
linear_model = LinearRegression()
linear_model.fit(X_linear, y_linear)

# Model regresji logistycznej
data_numeric['wysoki_opad'] = (data_numeric['suma_opadu'] > data_numeric['suma_opadu'].median()).astype(int)
X_logistic = data_numeric[['predkosc_wiatru', 'kierunek_wiatru', 'wilgotnosc_wzgledna', 'cisnienie']]
y_logistic = data_numeric['wysoki_opad']
logistic_model = LogisticRegression()
logistic_model.fit(X_logistic, y_logistic)

@app.route('/descriptive_stats', methods=['GET'])
def get_descriptive_stats():
    return descriptive_stats.to_json()

@app.route('/correlation', methods=['GET'])
def get_correlation():
    return correlation_matrix.to_json()

@app.route('/linear_regression', methods=['GET'])
def get_linear_regression():
    results = {
        'coefficients': linear_model.coef_.tolist(),
        'intercept': linear_model.intercept_
    }
    return jsonify(results)

@app.route('/logistic_regression', methods=['GET'])
def get_logistic_regression():
    results = {
        'coefficients': logistic_model.coef_.tolist(),
        'intercept': logistic_model.intercept_.tolist()
    }
    return jsonify(results)

@app.route('/linear_regression_plot', methods=['GET'])
def get_linear_regression_plot():
    plt.figure(figsize=(10, 6))
    sns.regplot(x=data_numeric['predkosc_wiatru'], y=data_numeric['temperatura'], line_kws={"color": "red"})
    plt.xlabel('Prędkość Wiatru')
    plt.ylabel('Temperatura')
    plt.title('Regresja Liniowa: Prędkość Wiatru vs. Temperatura')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

@app.route('/boxplot', methods=['GET'])
def get_boxplot():
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data_numeric)
    plt.title('Wykresy Pudełkowe dla Kolumn Numerycznych')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Ensure the labels do not get cut off

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
