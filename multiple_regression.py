from matplotlib import pyplot as plt
import random
from math import sqrt, log

def read_csv(file_name):
	with open(file_name, 'r') as f:
		data = []
		attribute_names = {index:name for index,name in enumerate(f.readline().strip().split(','))}
		while True:
			line = f.readline()
			if not line: return data
			else:
				data.append({attribute_names[index]:val for index,val in enumerate(line.strip().split(','))})

def simple_regression(input_feature, output):
	N = len(output)
	input_sum = sum(input_feature)
	input_squared_sum = sum([val**2 for val in input_feature])
	output_sum = sum(output)
	input_output_sum = sum([x*y for x,y in zip(input_feature, output)])
	slope = (input_output_sum - ((input_sum * output_sum) / N)) / (input_squared_sum - (input_sum ** 2 / N))
	intercept = (output_sum / N) - (slope * (input_sum / N))
	
	return(intercept, slope)

def get_regression_predictions(input_feature, intercept, slope):
	return [intercept + (slope * feature) for feature in input_feature]

def get_residual_sum_of_squares(feature_matrix, output, weights):
	# return sum([(y - (intercept + (x*slope))) ** 2 for x,y in zip(input_feature, output)])
	err = vector_subtract(output, matrix_vector_product(feature_matrix, weights))
	return dot(err,err)

def inverse_regression_predictions(output, intercept, slope):
	return (output - intercept) / slope


house_data = read_csv('kc_house_data.csv')

for point in house_data:
	point['bathrooms'] = float(point['bathrooms'])
	point['waterfront'] = int(point['waterfront'])
	point['sqft_above'] = int(point['sqft_above'])
	point['sqft_living15'] = float(point['sqft_living15'])
	point['grade'] = int(point['grade'])
	point['yr_renovated'] = int(point['yr_renovated'])
	point['price'] = float(point['price'])
	point['bedrooms'] = float(point['bedrooms'])
	point['zipcode'] = str(point['zipcode'])
	point['long'] = float(point['long'])
	point['sqft_lot15'] = float(point['sqft_lot15'])
	point['sqft_living'] = float(point['sqft_living'])
	point['floors'] = str(point['floors'])
	point['condition'] = int(point['condition'])
	point['lat'] = float(point['lat'])
	point['sqft_basement'] = int(point['sqft_basement'])
	point['yr_built'] = int(point['yr_built'])
	point['id'] = str(point['id'])
	point['sqft_lot'] = int(point['sqft_lot'])
	point['view'] = int(point['view'])

for point in house_data:
	point['bedrooms_squared'] = point['bedrooms'] ** 2
	point['bed_bath_rooms'] = point['bedrooms'] * point['bathrooms']
	point['log_sqft_living'] = log(point['sqft_living'])
	point['lat_plus_long'] = point['lat'] + point['long']

def predict_outcome(feature_matrix, weights):
	return [dot(row,weights) for row in feature_matrix]

def dot(v,w):
	return sum([v_i * w_i for v_i,w_i in zip(v,w)])

def magnitude(v):
	return sqrt(dot(v,v))

def vector_add(v,w):
	return [v_i + w_i for v_i,w_i in zip(v,w)]

def vector_subtract(v,w):
	return [v_i - w_i for v_i,w_i in zip(v,w)]

def matrix_vector_product(A,v):
	return [dot(A_i,v) for A_i in A]

def transpose(A):
	return [[row[col] for row in A] for col in range(len(A[0]))]

def scalar_vector_product(c,v):
	return [c*v_i for v_i in v]
			
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
	y = output
	w = initial_weights
	H = feature_matrix
	H_T = transpose(H)
	eta = step_size
	gradient_magnitude = float('inf')
	while tolerance < gradient_magnitude:
		
		RSS_gradient = scalar_vector_product(-2, matrix_vector_product(H_T, vector_subtract(y, matrix_vector_product(H,w))))
		# print w
		w = vector_subtract(w, scalar_vector_product(eta, RSS_gradient))
		gradient_magnitude = magnitude(RSS_gradient)
		# print RSS_gradient
		print gradient_magnitude

	return w

#def regression(feature_matrix, output):
	

random.seed(0)
random.shuffle(house_data)
train_data,test_data = (house_data[:int(len(house_data) * .8)],house_data[(int(len(house_data) * .8)):])

def simple_weights():
	simple_feature_matrix = [[1.0, point['sqft_living']] for point in train_data]
	output = [point['price'] for point in train_data]
	initial_weights = [-47000, 1.0]
	# step_size = 7e-12
	step_size = 7 * (10 ** -12) 
	tolerance = 2.5e7
	simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
	#print simple_weights
	return get_residual_sum_of_squares([[1.0, point['sqft_living']] for point in test_data], output, simple_weights)

def less_simple_weights():
	variable = 'bedrooms'
	degree = 5
	simple_feature_matrix = [[point[variable]**i for i in range(degree)] for point in train_data]
	output = [point['price'] for point in train_data]
	initial_weights = [1.0 for i in range(degree)]
	step_size = 1 * (10 ** -11)
	tolerance = 3e11
	simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
	print simple_weights
	# rss = get_residual_sum_of_squares([[1.0, point['bedrooms']] for point in test_data], output, simple_weights)

	plt.scatter([point[variable] for point in train_data], [point['price'] for point in train_data], s=1, alpha=.01)
	max_x = max([point[variable] for point in train_data])
	max_y = max([point['price'] for point in train_data])
	xs = []
	ys = []
	segment_count = 1000
	for i in range(segment_count):
		x = i*(10.0 / segment_count)
		xs.append(x)
		y = dot([x**i for i in range(degree)], simple_weights)
		#print x,y
		ys.append(y)
		# ys.append(500000.0)
	plt.plot(xs, ys, 'k-')
	
	plt.xlim(0,max_x)
	plt.ylim(0,max_y)
	
	plt.xlabel(variable)
	plt.ylabel('price USD')
	plt.title(variable+' vs. price')
	plt.show()


#print simple_weights()
train_data = [point for point in train_data if point['bedrooms'] != 33]
less_simple_weights()

	
# intercept,slope = simple_regression(sqft_vals, price_vals)
# print 'attributes: ', [attr for attr in train_data[0].keys()] 
# print intercept,slope
# print get_regression_predictions([2650], intercept, slope)
# 
# plt.scatter([point['sqft_living'] for point in train_data], output)
# plt.plot([0, 14000], [simple_weights[0], simple_weights[1] * 14000], 'k-')
# 
# plt.ylim(0,max(output))
# # plt.xlim(0,max(sqft_vals))
# 
# plt.xlabel('sqft')
# plt.ylabel('price USD')
# plt.title('sqft vs. price')
# plt.show()
