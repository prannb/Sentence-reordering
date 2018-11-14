import numpy as np
import random

sample_pred = np.array([3,1,4,5,2,0])
sample_actual = np.array([3,4,1,2,5,0])
# def actualMatrix(matrix):



# def mod_distance(pred, actual):

def correct_pairwise(pred, actual):

	l_pred = list(pred)
	l_actual = list(actual)

	n = len(pred)
	total_pairs = (n*(n-1))/2

	correct_placement = 0

	for i in range(n):
		for j in range(i+1,n):
			if l_pred.index(l_actual[i]) < l_pred.index(l_actual[j]):
				correct_placement += 1
	return correct_placement * 1.0 / total_pairs

def correct_placements(pred, actual):

	l_pred = list(pred)
	l_actual = list(actual)

	n = len(pred)

	correct_placement = 0

	for i in range(n):
		if(l_pred[i] == l_actual[i]):
			correct_placement += 1
	return correct_placement * 1.0 / n

def mod_diff(pred, actual):
	l_pred = list(pred)
	l_actual = list(actual)
	random.shuffle(l_actual)

	n = len(pred)

	dev = 0


	for i in range(n):
		dev += abs(l_pred[i] - l_actual[i])
	return dev


print(correct_pairwise(sample_pred, sample_actual))
print(correct_placements(sample_pred, sample_actual))

mod_diff(sample_pred, sample_actual)