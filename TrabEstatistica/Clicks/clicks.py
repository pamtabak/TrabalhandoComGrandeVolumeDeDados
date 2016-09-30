import matplotlib as plt
import numpy as np
import math
import scipy.stats as stats

import csv

######### Lendo arquivos csv #########

telaSemPopup          = []
csvfileTelaSemPopup   = open('amostra_A_click.csv', 'rb')
csvreaderTelaSemPopup = csv.reader(csvfileTelaSemPopup, delimiter=';')

# Tirando o cabecalho
csvreaderTelaSemPopup.next()

for row in csvreaderTelaSemPopup:
	telaSemPopup.append(row[1])


telaComPopup          = []
csvfileTelaComPopup   = open('amostra_B_click.csv', 'rb')
csvreaderTelaComPopup = csv.reader(csvfileTelaComPopup, delimiter=';')

# Tirando o cabecalho
csvreaderTelaComPopup.next()

for row in csvreaderTelaComPopup:
	telaComPopup.append(row[1])


######### Calculando valores observados #########
cliqueTelaSemPopupObservado    = telaSemPopup.count("yes")              # X
naoCliqueTelaSemPopupObservado = telaSemPopup.count("no")               # W
cliqueTelaComPopupObservado    = telaComPopup.count("yes")              # Y
naoCliqueTelaComPopupObservado = telaComPopup.count("no")               # Z
totalObservado                 = len(telaSemPopup) + len(telaComPopup)  # T


######### Calculando valores esperados #########
cliqueTelaSemPopupEsperado    = ((cliqueTelaSemPopupObservado + cliqueTelaComPopupObservado) * (naoCliqueTelaSemPopupObservado + cliqueTelaSemPopupObservado)) / totalObservado
naoCliqueTelaSemPopupEsperado = ((naoCliqueTelaSemPopupObservado + naoCliqueTelaComPopupObservado) * (naoCliqueTelaSemPopupObservado + cliqueTelaSemPopupObservado)) / totalObservado
cliqueTelaComPopupEsperado    = ((cliqueTelaSemPopupObservado + cliqueTelaComPopupObservado) * (cliqueTelaComPopupObservado + naoCliqueTelaComPopupObservado)) / totalObservado
naoCliqueTelaComPopupEsperado = ((naoCliqueTelaSemPopupObservado + naoCliqueTelaComPopupObservado) * (naoCliqueTelaComPopupObservado + cliqueTelaComPopupObservado)) / totalObservado

######### Calculando a estatistica #########
xn  = math.pow(cliqueTelaSemPopupObservado    - cliqueTelaSemPopupEsperado, 2)    / cliqueTelaSemPopupEsperado
xn += math.pow(naoCliqueTelaSemPopupObservado - naoCliqueTelaSemPopupEsperado, 2) / naoCliqueTelaSemPopupEsperado
xn += math.pow(cliqueTelaComPopupObservado    - cliqueTelaComPopupEsperado, 2)    / cliqueTelaComPopupEsperado
xn += math.pow(naoCliqueTelaComPopupObservado - naoCliqueTelaComPopupEsperado, 2) / naoCliqueTelaComPopupEsperado

numeroDeGrausDeLiberdade = 4
alfa = 0.05
pvalue = 1 - stats.chi2.cdf(xn, numeroDeGrausDeLiberdade) #1 - acumulada ate o ponto xn
print (pvalue)

if pvalue > alfa:
	print ("Nao rejeita H0. Popup nao faz diferenca")
else:
	print ("Rejeita H0. Popup faz diferenca")

