import matplotlib as plt
import numpy as np
import math
import scipy.stats as stats

import csv

# H0 : m0 = mediaPopulacao
# H1 : Rejeita H0 (m0 != mediaPopulacao)
# sendo m0 a media da amostra. 

# O objetivo e descobrir se vale a pena implementar a feature (caso melhore) 
# ou remove-la (caso ocorra uma piora)

# Lendo arquivo csv da populacao
tempoPopulacao     = []
csvfilePopulacao   = open('populacao_tempo.csv', 'rb')
csvreaderPopulacao = csv.reader(csvfilePopulacao, delimiter=';')

# Tirando o cabecalho
csvreaderPopulacao.next()

for row in csvreaderPopulacao:
	tempoPopulacao.append(float(row[1]))	

# Calcular media e desvio da populacao
mediaPopulacao        = np.mean(np.array(tempoPopulacao))
desvioPadraoPopulacao = math.sqrt(np.var(np.array(tempoPopulacao)))	
tamanhoPopulacao      = len(tempoPopulacao)

# Lendo arquivo csv da amostra
tempoAmostra     = []
csvfileAmostra   = open('amostra_tempo.csv', 'rb')
csvreaderAmostra = csv.reader(csvfileAmostra, delimiter=';')

# Tirando o cabecalho
csvreaderAmostra.next()

for row in csvreaderAmostra:
	tempoAmostra.append(float(row[1]))

numeroDeAmostras = len(tempoAmostra)

mediaAmostra = np.mean(np.array(tempoAmostra))

# Teste de Hipotese (Bicaudal)
alfa = 0.05

z = (mediaAmostra - mediaPopulacao)/ (desvioPadraoPopulacao / math.sqrt(tempoPopulacao))

# ponto critico: P[Z > z]
pValor - 0
if (z > 0):
	pValor = 1 - stats.norm.cdf(z)
else:
	pValor = stats.norm.cdf(z)

if (pValor < alfa/2):
	if (z > 0):
		print("Aceita a New Feature - Rejeita H0")
	else:
		print("Rejeita a New Feature - Rejeita H0")
else:
	print ("Nao Rejeita H0")