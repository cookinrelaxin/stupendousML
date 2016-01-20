from matplotlib import pyplot as plt

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

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
	return sum([(y - (intercept + (x*slope))) ** 2 for x,y in zip(input_feature, output)])

def inverse_regression_predictions(output, intercept, slope):
	return (output - intercept) / slope

house_data = read_csv('kc_house_data.csv')
train_data,test_data = (house_data[:int(len(house_data) * .8)],house_data[int(len(house_data) * .8):])

sqft_vals = [float(point['sqft_living']) for point in train_data]
price_vals = [float(point['price']) for point in train_data]

intercept,slope = simple_regression(sqft_vals, price_vals)
print 'attributes: ', [attr for attr in train_data[0].keys()] 
print intercept,slope
print get_regression_predictions([2650], intercept, slope)

plt.scatter(sqft_vals, price_vals)
plt.plot([0, 14000], [intercept, slope * 14000], 'k-')

plt.ylim(0,max(price_vals))
plt.xlim(0,max(sqft_vals))

plt.xlabel('sqft')
plt.ylabel('price USD')
plt.title('sqft vs. price')
plt.show()

