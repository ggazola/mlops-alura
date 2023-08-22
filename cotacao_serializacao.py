from flask import Flask, request, jsonify

# importação do basic-auth do flask
import flask_basicauth
from flask_basicauth import BasicAuth

# importar a biblioteca
from textblob import TextBlob

# importando bibliotecas para machine learning
from sklearn.linear_model import LinearRegression  # modelo regressão linear

# importar o pickle dentro da nossa API
import pickle

# importar ambiente
import os

# ler o arquivo 'modelo.sav'
modelo = pickle.load(open('../../models/modelo.sav', 'rb') )

# forçar a entrada dos dados
colunas = ['tamanho', 'ano', 'garagem']  # parsing dos dados

#app = Flask('meu_app')
app = Flask('__name__')

# configuração para autenticação (login e senha)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

#  garantir que você só faço o acesso dessa API passando essa autenticação básica
basic_auth = BasicAuth(app)

# definição das rotas
@app.route('/')  # rota/endereço base

# endpoint de 'sentimentos' para fazer um teste
@app.route('/sentimento/<frase>')
@basic_auth.required

def analisar_sentimento(frase):
    resultado = sentimento(frase)
    return resultado

# função para pegar o sentimento
def sentimento(frase):
    # o que acontece quando entra uma frase, o que retornamos.
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt', to='en')

    # retornar a polaridade
    polaridade = tb_en.sentiment.polarity

    return 'Polaridade: {}'.format(polaridade)

# novo endpoint para avaliar e prever preço das casas
@app.route('/cotacao/', methods=['POST'])
@basic_auth.required

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
app.run(debug=True, host='0.0.0.0')