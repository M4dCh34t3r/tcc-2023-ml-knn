# Importacao de bibliotecas
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from os import path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys

# Declaracao de constantes
DADOS_COLUNAS_USADAS = [
  "drinks_alcohol",
  "gets_drunk",
  "is_overweight",
  "use_marijuana",
  "understanding_parents",
  "missed_classes",
  "has_sexual_relation",
  "smoke_cigarettes",
  "had_fights",
  "is_bullied",
  "got_seriously_injured",
  "no_close_friends",
  "attempted_suicide"
]
NOME_ARQUIVO = 'ghsh_suicidal_behaviors_teenagers.csv'

# Verificando a existencia do arquivo
if not path.exists(NOME_ARQUIVO):
    sys.exit(
        '''
        O arquivo com os dados nao consta no diretorio do projeto, por favor
        importe-o e adicione nesta mesma pasta, segue o link de acesso ao arquivo
        https://drive.google.com/file/d/1iYUGzjHHKIXmE_qaAzSTXO7iv5xmZKpa/view?usp=sharing
        '''
    )

# Importando arquivo e listando valores
dados = pd.read_csv(NOME_ARQUIVO)
dados_colunas = list(dados.columns.values)

# Iniciando a analise dos dados
print(f'Os dados contidos no "dataframe" correspondem à...')
for coluna in dados_colunas:
  print(coluna)

print(
    f"\nMédia de envolvidos que consomem álcool - { round(np.mean(dados['drinks_alcohol']), 2) }%" +
    f"\nMédia de envolvidos que se embebedam - { round(np.mean(dados['gets_drunk']), 2) }%" +
    f"\nMédia de envolvidos que são acima do peso - { round(np.mean(dados['is_overweight']), 2) }%" +
    f"\nMédia de envolvidos que fumam maconha - { round(np.mean(dados['use_marijuana']), 2) }%" +
    f"\nMédia de envolvidos que possuem pais compreensivos - { round(np.mean(dados['understanding_parents']), 2) }%" +
    f"\nMédia de envolvidos que faltam aulas - { round(np.mean(dados['missed_classes']), 2) }%" +
    f"\nMédia de envolvidos que praticam atividades sexuais - { round(np.mean(dados['has_sexual_relation']), 2) }%" +
    f"\nMédia de envolvidos que fumam cigarros - { round(np.mean(dados['smoke_cigarettes']), 2) }%" +
    f"\nMédia de envolvidos que se envolveram em brigas - { round(np.mean(dados['had_fights']), 2) }%" +
    f"\nMédia de envolvidos que sofrem bullying - { round(np.mean(dados['is_bullied']), 2) }%" +
    f"\nMédia de envolvidos que se machucaram sériamente - { round(np.mean(dados['got_seriously_injured']), 2) }%" +
    f"\nMédia de envolvidos que não possuem amigos proximos - { round(np.mean(dados['no_close_friends']), 2) }%" +
    f"\nMédia de envolvidos que tentaram suicídio - { round(np.mean(dados['attempted_suicide']), 2) }%" 
)

x = dados[DADOS_COLUNAS_USADAS].values
y = LabelEncoder().fit_transform(dados['sex'].values)

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.5,random_state=0)

# Lista para testes de melhor número de vizinhos próximos
vizinhos_possiveis_min_max = list(range(1, 10))

# Armazenando testes de correspondencia para o KNN com os valores anteriores
cv_scores = []
for k in vizinhos_possiveis_min_max:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,x_treino,y_treino,cv=10,scoring='accuracy')
    cv_scores.append(scores.mean())

# Erro Quadrático Médio
eqm = [1 - x for x in cv_scores]

# Informando a melhor correspondência para vizinhos próximos
melhor_qtde_vizinhos = vizinhos_possiveis_min_max[eqm.index(min(eqm))]
print(f'\nA quantidade ideal de vizinhos foi { melhor_qtde_vizinhos }')

# Treinando algoritmo com o melhor valor encontrado
classifier = KNeighborsClassifier(n_neighbors=melhor_qtde_vizinhos)
classifier.fit(x_treino, y_treino)

# Prevendo os dados com base no algoritmo
y_previs = classifier.predict(x_teste)

# Testando a acurácia do modelo
precisao = accuracy_score(y_teste, y_previs) * 100
print(f'\nA acurácia aproximada do modelo foi { round(precisao,2) }%')

print(f'\nGerando gráficos para visualização dos dados...')

# Plotando gráfico com os dados previstos
sns.pairplot(data=dados,height=2,hue='sex')
plt.show()

# Plotando gráfico de correspondência de vizinhos próximos
plt.figure(figsize=(6,4),num="Gráfico de correspondência de vizinhos próximos")
plt.xlabel('Qtde estimada de vizinhos',fontsize=8)
plt.ylabel('Erro quadrático médio',fontsize=8)
plt.title('Qtde ideal de vizinhos',fontsize=16,fontweight='bold')
plt.plot(vizinhos_possiveis_min_max,eqm)
plt.show()

# Plotando mapa de calor
sns.heatmap(confusion_matrix(y_teste, y_previs),annot=True)
plt.xlabel("Valor previsto")
plt.ylabel("Valor verdadeiro")
plt.show()
