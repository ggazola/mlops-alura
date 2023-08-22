from flask import Flask

# importar a biblioteca
from textblob import TextBlob

# importando bibliotecas para machine learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # modelo regressão linear

# lendo o dataset
df = pd.read_csv('dados/casas.csv')

# redução do dataframe
colunas = ['tamanho' , 'preco']
df = df[colunas]

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

# nova rota
@app.route('/sentimento/<frase>')
def analisar_sentimento(frase):
    resultado = sentimento(frase)
    return resultado

# definir função a ser executada quando chegar na rota
def home():
    return 'Minha primeira API.'

# função para pegar o sentimento
def sentimento(frase):
    # o que acontece quando entra uma frase, o que retornamos.
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt', to='en')

    # retornar a polaridade
    polaridade = tb_en.sentiment.polarity

    return 'Polaridade: {}'.format(polaridade)

# novo endpoint para avaliar e prever preço das casas
@app.route('/cotacao/<int:tamanho>')
def cotacao(tamanho):
    
    # vamos passar esse “predict” dentro dessas, uma lista dentro da outra
    preco = modelo.predict([[tamanho]])
    return str(preco)

# enviar o script para execução
# app.run()  # modo debug off
app.run(debug=True)