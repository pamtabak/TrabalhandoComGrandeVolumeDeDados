import matplotlib as plt
import numpy as np
import math
import scipy.stats as stats

import csv

#Lendo o arquivo de texto
csvFile   = open('movie_metadata.csv', 'rb')
csvReader = csv.reader(csvFile, delimiter=',')

# Tirando o cabecalho
csvReader.next()

#facenumber_in_poster = index 15
#imdb_score           = index 25

faceNumber = []
imdbScore  = []

for row in csvReader:
	if (row[15] != '' and row[25] != ''):
		#Garantindo que os dados (com os quais queremos trabalhar) estao preenchidos
		faceNumber.append(float(row[15]))
		imdbScore.append(float(row[25]))

#Checando a correlacao entre as duas variaveis
correlation = stats.pearsonr(faceNumber, imdbScore)
print(correlation)