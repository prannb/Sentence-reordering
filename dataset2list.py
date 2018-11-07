import csv
import pickle
def create_list(filename = 'data/training_set_rel3.tsv'):
	with open(filename, 'r') as tsv:
		print(tsv)
		a = [line.strip().split('\t') for line in tsv]

	dataset = []

	# for ele in a:
	# 	dataset.append(ele[2][1:-1])
	return dataset
filename = "train_list"
file_obj = open(filename, 'wb')
pickle.dump(create_list(), file_obj)