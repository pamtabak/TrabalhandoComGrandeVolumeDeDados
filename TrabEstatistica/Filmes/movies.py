import matplotlib as plt
import numpy as np
import math
import scipy.stats as stats

import csv

#Lendo o arquivo de texto
csvFile   = open('movie_metadata.csv', 'rb')
csvReader = csv.reader(csvFile, delimiter=',')

#Tentando descobrir a melhor correlacao entre imdbScore e outra variavel
#de modo a ajudar o produtor

headers   = [] #array com o cabecalho
atributos = [] #array de dimensao 2, com 28 (numero de campos) posicoes. Cada array interno tera o numero de linhas do dataset - 1 (remove o cabecalho)

#Indices numericos
indices=[2,3,4,5,7,8,12,13,15,18,22,23,24,25,26,27]

#28 atributos
row = csvReader.next()
for x in indices:
	headers.append(row[x])
	cadaAtributo = []
	atributos.append(cadaAtributo)

for row in csvReader:
	linhaEstaCompleta = True
	for x in indices:
		if (row[x] == ''):
			linhaEstaCompleta = False
			break
	if (linhaEstaCompleta):
		for x in indices:
			atributos[indices.index(x)].append(row[x])

#Calculando a correlacao entre imdb_score e facenumber_in_poster
faceNumber = np.array(atributos[indices.index(15)]).astype(np.float)
imdbScore  = np.array(atributos[indices.index(25)]).astype(np.float)

correlacao = stats.pearsonr(faceNumber, imdbScore)
print("Correlacao entre imdb_score e facenumber_in_poster = " + str(correlacao))

#Calculando a correlacao entre imdb_score e outra variavel numerica
maiorCorrelacao = 0
indiceMaiorCorrelacao = 0

for x in indices:
	if x == 25: #propria variavel imdb_score
		continue
	correlacao = stats.pearsonr(np.array(atributos[indices.index(x)]).astype(np.float), imdbScore)
	if (correlacao > maiorCorrelacao):
		maiorCorrelacao = correlacao
		indiceMaiorCorrelacao = x
	print(str(correlacao[0]) + " " + headers[indices.index(x)])

print("A maior correlacao encontrada para imdb_score com alguma variavel (numerica) = " + str(maiorCorrelacao[0]) + ". Variavel = " + headers[indices.index(indiceMaiorCorrelacao)])


