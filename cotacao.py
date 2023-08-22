from flask import Flask, request, jsonify

# importar a biblioteca
from textblob import TextBlob

# importando bibliotecas para machine learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # modelo regressão linear

# lendo o dataset
df = pd.read_csv('dados/casas.csv')

# forçar a entrada dos dados
colunas = ['tamanho', 'ano', 'garagem']  # parsing dos dados

# separação das variávies (explicativas e resposta)
X = df.drop('preco', axis=1 )
y = df['preco']

# separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)

# ajuste do modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

#app = Flask('meu_app')
app = Flask('__name__')

# definição das rotas
@app.route('/')  # rota/endereço base

# novo endpoint para avaliar e prever preço das casas
@app.route('/cotacao/', methods=['POST'])
def cotacao():
    # o que o usuário enviar irá armazenar o formato JSON na variável dados
    dados = request.get_json()

    # criar uma lista (list comprehension)
    dados_input = [ dados[col] for col in colunas ]

    # vamos passar esse “predict” dentro de uma lista 
    preco = modelo.predict([dados_input])
    return jsonify(preco = preco[0])

# enviar o script para execução
# app.run()  # modo debug off
app.run(debug=True)