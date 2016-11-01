import matplotlib as plt
import numpy as np
import csv

#import math
#import scipy.stats as stats

# Target is the real class of the object (last attribute)
# Target 1 is the less likely (and Target 0 the most likely)

# Reading dataset
csvFile   = open('dataset_trabalho3/train_file.csv', 'rb')
csvReader = csv.reader(csvFile, delimiter=',')

# Array of arrays. Each array is a column of the dataset
dataset = []
header  = []

# Reading headers (so we know the amount of columns there are)
row = csvReader.next()
for column in row:
	header.append(column)
	dataset.append([])

for row in csvReader:
	col = 0
	for column in row:
		dataset[col].append(column)
		col += 1

# Checking what we can do with each column
# 1. If there is any column that doesn`t make any difference (for instance, all the records are filled with the same value)
col = 0
for column in dataset:
	allElementsEqual = True
	value = column[0]
	for x in column:
		if (x != value):
			allElementsEqual = False
			break
	if (allElementsEqual):
		header.remove(header[col])
		dataset.remove(column)
	col += 1

