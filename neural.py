import math
import random
import string
import os
import csv

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
	return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
	m = []
	for i in range(I):
		m.append([fill]*J)
	return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
	return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
	return 1.0 - y**2

class NN:
	def __init__(self, ni, nh, no):
		# number of input, hidden, and output nodes
		self.ni = ni + 1 # +1 for bias node
		self.nh = nh
		self.no = no

		# activations for nodes
		self.ai = [1.0]*self.ni
		self.ah = [1.0]*self.nh
		self.ao = [1.0]*self.no
		
		# create weights
		self.wi = makeMatrix(self.ni, self.nh)
		self.wo = makeMatrix(self.nh, self.no)
		# set them to random vaules
		for i in range(self.ni):
			for j in range(self.nh):
				self.wi[i][j] = rand(-0.5, 0.5)
		for j in range(self.nh):
			for k in range(self.no):
				self.wo[j][k] = rand(-3.0, 3.0)

		# last change in weights for momentum   
		self.ci = makeMatrix(self.ni, self.nh)
		self.co = makeMatrix(self.nh, self.no)

	def update(self, inputs):
		if len(inputs) != self.ni-1:
			raise ValueError('wrong number of inputs')

		# input activations
		for i in range(self.ni-1):
			#self.ai[i] = sigmoid(inputs[i])
			self.ai[i] = inputs[i]

		# hidden activations
		for j in range(self.nh):
			sum = 0.0
			for i in range(self.ni):
				sum = sum + self.ai[i] * self.wi[i][j]
			self.ah[j] = sigmoid(sum)

		# output activations
		for k in range(self.no):
			sum = 0.0
			for j in range(self.nh):
				sum = sum + self.ah[j] * self.wo[j][k]
			self.ao[k] = sigmoid(sum)

		return self.ao[:]


	def backPropagate(self, targets, N, M):
		if len(targets) != self.no:
			raise ValueError('wrong number of target values')

		# calculate error terms for output
		output_deltas = [0.0] * self.no
		for k in range(self.no):
			error = targets[k]-self.ao[k]
			output_deltas[k] = dsigmoid(self.ao[k]) * error

		# calculate error terms for hidden
		hidden_deltas = [0.0] * self.nh
		for j in range(self.nh):
			error = 0.0
			for k in range(self.no):
				error = error + output_deltas[k]*self.wo[j][k]
			hidden_deltas[j] = dsigmoid(self.ah[j]) * error

		# update output weights
		for j in range(self.nh):
			for k in range(self.no):
				change = output_deltas[k]*self.ah[j]
				self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
				self.co[j][k] = change
				#print N*change, M*self.co[j][k]

		# update input weights
		for i in range(self.ni):
			for j in range(self.nh):
				change = hidden_deltas[j]*self.ai[i]
				self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
				self.ci[i][j] = change

		# calculate error
		error = 0.0
		for k in range(len(targets)):
			error = error + 0.5*(targets[k]-self.ao[k])**2
		return error


	def test(self, patterns):
		for p in patterns:
			print(p[0], '->', self.update(p[0]))

	def weights(self):
		print('Input weights:')
		for i in range(self.ni):
			print(self.wi[i])
		print()
		print('Output weights:')
		for j in range(self.nh):
			print(self.wo[j])

	def train(self, patterns, iterations=300000, N=0.03, M=0.03):
		# N: learning rate
		# M: momentum factor
		for i in range(iterations):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.update(inputs)
				error = error + self.backPropagate(targets, N, M)
			if i % 100 == 0:
				print('error %-.10f' % error)


def demo():
	sample = []
	sample2 = []
	num_sample = 0
	rootDir = './'
	consumption = 0
	
	
	for c_dirName, c_subdirList, c_fileList in os.walk(rootDir+'consumption'):
		i = 0
		for c_fname in c_fileList:
			f = open(rootDir+'consumption'+'/'+c_fname, 'r',encoding='big5')
			f.readline()
			consumption = 0
			for line in f.readlines():
				if float(line.split(' ')[0]) > 1000:
					consumption += float(line.split(' ')[0])
			
			for t_dirName, t_subdirList, t_fileList in os.walk(rootDir+'temperature'):
				for t_fname in t_fileList:
					if c_fname == t_fname:
						f = open(rootDir+'temperature'+'/'+t_fname, 'r',encoding='utf-8')
						spamreader = csv.reader(f, delimiter=',', quotechar='"')
						Templist = [float(row[0])for row in spamreader]
				
						try:
							latestAvgTemp = sum(Templist) / len(Templist)
							latestMinTemp = min(Templist)
							latestMaxTemp = max(Templist)
						except ZeroDivisionError:
							continue
							
			for h_dirName, h_subdirList, h_fileList in os.walk(rootDir+'humidity'):
				for h_fname in h_fileList:
					if c_fname == h_fname:
						f = open(rootDir+'humidity'+'/'+h_fname, 'r',encoding='utf-8')
						spamreader = csv.reader(f, delimiter=',', quotechar='"')
						Humiditylist = [float(row[0])for row in spamreader]
						try:
							latestAvgHumidity = sum(Humiditylist) / len(Humiditylist)
							latestMinHumidity = min(Humiditylist)
							latestMaxHumidity = max(Humiditylist)
						except ZeroDivisionError:
							continue
							
			for i_dirName, i_subdirList, i_fileList in os.walk(rootDir+'irradiance'):
				for i_fname in i_fileList:
					if c_fname == i_fname:
						f = open(rootDir+'irradiance'+'/'+i_fname, 'r',encoding='utf-8')
						spamreader = csv.reader(f, delimiter=',', quotechar='"')
						Irradiancelist = [float(row[0])for row in spamreader]
						try:
							latestAvgIrradiance = sum(Irradiancelist) / len(Irradiancelist)
							latestMinIrradiance = min(Irradiancelist)
							latestMaxIrradiance = max(Irradiancelist)
						except ZeroDivisionError:
							continue
			
			sample.append([[latestAvgTemp, latestMaxTemp, latestMinTemp, latestAvgHumidity, latestMaxHumidity, latestMinHumidity, latestAvgIrradiance], [consumption]])
			sample2.append([[latestAvgTemp, latestMaxTemp, latestAvgHumidity, latestMaxHumidity], [consumption]])
			
			num_sample += 1
			
		#normalization	
		avgalist = []
		minalist = []
		maxalist = []
		avghlist = []
		minhlist = []
		maxhlist = []
		conslist = []
		avgilist = []
		
		for [[avgt,maxt,mint,avgh,maxh,minh,avgi],[cons]] in sample:
			#溫度
			avgalist.append(avgt)
			maxalist.append(maxt)
			minalist.append(mint)
			#濕度
			avghlist.append(avgh)
			maxhlist.append(maxh)
			minhlist.append(minh)
			#日照
			avgilist.append(avgi)
			#用電量
			conslist.append(cons)
			
		
		min_temp = min(maxalist)
		max_temp = max(minalist)
		
		min_humidity = min(maxhlist)
		max_humidity = max(minhlist)
		
		max_consu = max(conslist)
		min_consu = min(conslist)
		
		max_avgi = max(avgilist)
		min_avgi = min(avgilist)
		
		i = 0
		while i < num_sample:
			'''
			sample[i][0][0] = 2 *(sample[i][0][0] - min_temp) / (max_temp - min_temp) - 1
			sample[i][0][1] = 2 *(sample[i][0][1] - min_temp) / (max_temp - min_temp) - 1
			sample[i][0][2] = 2 *(sample[i][0][2] - min_temp) / (max_temp - min_temp) - 1
			sample[i][0][3] = 2 *(sample[i][0][3] - min_humidity) / (max_humidity - min_humidity) - 1
			sample[i][1][0] = 2 *(sample[i][1][0] - min_consu) / (max_consu - min_consu) - 1
			'''
			sample2[i][0][0] = 2 *(sample2[i][0][0] - min_temp) / (max_temp - min_temp) - 1
			sample2[i][0][1] = 2 *(sample2[i][0][1] - min_temp) / (max_temp - min_temp) - 1
			sample2[i][0][2] = 2 *(sample2[i][0][2] - min_humidity) / (max_humidity - min_humidity) - 1
			sample2[i][0][3] = 2 *(sample2[i][0][3] - min_humidity) / (max_humidity - min_humidity) - 1
			sample2[i][1][0] = 2 *(sample2[i][1][0] - min_consu) / (max_consu - min_consu) - 1
			
			i += 1
			
			
	# 建置3個輸入 3個隱藏 1個輸出 之神經網路

	train = sample2[:-3]
	test = sample2[-5:]
	
	# create a network with two input, two hidden, and one output nodes
	n = NN(4, 4, 1)
	# train it with some patterns
	n.train(train)
	# test it
	n.test(test)
	#n.weights()
	print()
	for i in test:
		print(i[1][0])
	
if __name__ == '__main__':
	demo()