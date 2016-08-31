#Dentre os personagens de GOT que ainda nao morreram, qual a porcentagem dos que apareceram em todos os livros?

import csv

liveCharacters = []

#python 2 need to open the file as a binary file (rb):
csvfile = open('character-deaths.csv', 'rb')
csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
for row in csvreader:
	yearOfDeath = row[2]
	if (yearOfDeath == ""):
		# character is alive
	# print(yearOfDeath)
	# liveCharacters.append(row)