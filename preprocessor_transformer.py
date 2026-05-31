import numpy as np 


# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size, pred_size):
	data_X,data_Y = [], []
	for i in range(len(dataset)-step_size-pred_size):
		data_X.append(dataset[i:(i+step_size), 0])
		data_Y.append(dataset[i+step_size:i+step_size+pred_size, 0])
	return np.array(data_X), np.array(data_Y)
# THIS FUNCTION CAN BE USED TO CREATE A TIME SERIES DATASET FROM ANY 1D ARRAY	