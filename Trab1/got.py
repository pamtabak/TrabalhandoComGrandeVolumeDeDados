#Dentre os personagens de GOT que ainda nao morreram, qual a porcentagem dos que apareceram em todos os livros?

import csv

charactersAlive           = 0
charactersAliveInAllBooks = 0

#Ps: For some reason (it seems mac osx problems), the csv file was only recognized when adding a pipe (|) to the end of eachline
#Because of that, every argument regarding whether character appared in 5th book or not has a pipe at the end
#We need to clean this in order to use this information

#Reading file and selecting only valid data (characters that are still alive)
#python 2 need to open the file as a binary file (rb):
csvfile = open('character-deaths.csv', 'rb')
csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
for row in csvreader:
	yearOfDeath = row[2]
	if (yearOfDeath == ""):
		#character is alive
		charactersAlive += 1
		#checking if character has appeared in all books
		if (row[8] == "1" and row[9] == "1" and row[10] == "1" and row[11] == "1" and row[12].replace("|","") == "1"):
			charactersAliveInAllBooks += 1

#We multiply the numerator by 1.0 in order to make it a float and generate a float result (not only the integer part of the result)
#Multiplying whole result by 100 just to appear like a percentage
percentage = 100 * (charactersAliveInAllBooks * 1.0/ charactersAlive)
print percentage,"%"
