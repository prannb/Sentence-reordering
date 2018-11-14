import csv
import pickle
import pandas as pd
def create_list(filename = 'data/python3_data.tsv'):
	with open(filename, 'r') as tsv:
		print(tsv)
		a = [line.strip().split('\t') for line in tsv]

	dataset = []

	for ele in a:
		dataset.append(ele[2][1:-1])
	return dataset
# filename = "train_list"
# file_obj = open(filename, 'wb')
# data = create_list()
# pickle.dump(data, file_obj)
# print(data)


def create_list_nips(filename = 'data/papers.csv'):
	df = pd.read_csv(filename)
	dataset = []
	for ele in df.values:
		if ele[5] == 'Abstract Missing':
			continue
		else:
			dataset.append(ele[5])

	return dataset