# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:49:25 2019
@author: palan
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import os
#from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

# Importing the dataset
dirpath = os.getcwd()
print("current directory is : " + dirpath)
filepath = " "

#If no filepath given as input
filepath = input("Enter file path(If the file is kept on run directory press Enter twice):")
#filepath = r'C:\Users\palan\OneDrive\Desktop\Code - Github\machinelearning\Clemson- Machine Learning'
if not filepath.strip():
	filepath = dirpath
#If no Filename given as input, Filename is fixed and provided
filename = input('Enter File Name : ')
if not filename.strip():
	filename = "new_data_set.csv"

#  filename = filepath+"\\"+filename
#Total count of records
#datafile= pd.read_csv(filename,header = None)
#tot= int(datafile.loc[0,0])
tot = 82912
#Rest of the file
datafile= pd.read_csv(filename,header=None,skiprows=[0], sep=',')
#print(datafile)
#Shuffle the file
datafile = datafile.reindex(np.random.permutation(datafile.index))
split_point = int(tot*0.7)
train_data,test_data = datafile[:split_point], datafile[split_point:]
inp_data_train = train_data.iloc[:, [0,1,2,3,4]].values
pred_train = train_data.iloc[:, -1].values
inp_data_test = test_data.iloc[:, [0,1,2,3,4]].values
pred_test = test_data.iloc[:, -1].values
#print('length',len(inp_data_train),'inp_data_train',inp_data_train,'pred_train',pred_train,'inp_data_test',inp_data_test,'pred_test',pred_test)
#train data
inp_data_train_x0 = inp_data_train[:,0]
inp_data_train_x1 = inp_data_train[:,1]
inp_data_train_x2 = inp_data_train[:,2]
inp_data_train_x3 = inp_data_train[:,3]
inp_data_train_x4 = inp_data_train[:,4]
#test data
inp_data_test_x0 = inp_data_test[:,0]
inp_data_test_x1 = inp_data_test[:,1]
inp_data_test_x2 = inp_data_train[:,2]
inp_data_test_x3 = inp_data_train[:,3]
inp_data_test_x4 = inp_data_train[:,4]

iterations = 10000
alpha = 0.001

w0= 1#1.3#0.5,0.0009,0.003,0.00005,0.00000132,0.00001515
w1 = 0.1#1.2
w2 = 0.5#.0002
w3 = 1#.00025
w4 = 0.2#.3
w5 = 0#.00015

############################################################################
#def normalize(inp_data_train,feature_index):
#	result = np.empty(len(inp_data_train[:,0]))
#	max_value = inp_data_train[:,feature_index].max()
#	#print('max_value',max_value)
#	min_value = inp_data_train[:,feature_index].min()
#	#print('min_value',min_value)
#	result = (inp_data_train[:,feature_index] - min_value) / (max_value - min_value)
#	return result
#inp_data_train_x1 = normalize(inp_data_train,0)
#inp_data_train_x2 = normalize(inp_data_train,1)
#inp_data_train = np.concatenate([(inp_data_train_x1,inp_data_train_x2)])
#print('inp_data_train_normalized',inp_data_train)
#####################################
fig = plt.figure(figsize = (20,10))
ax=fig.add_subplot(111,projection="3d")
plt.title('Raw Data')
ax.scatter(inp_data_train_x2, inp_data_train_x3, pred_train, c='r',marker='o')#, cmap='hsv')
ax.set_xlabel('Floor')
ax.set_ylabel('Temperature')
ax.set_zlabel('Power')
plt.show()
#################################################################
# Calculate the mean value of a list of numbers
def mean_c(inp_data_train_x1):
	return sum(inp_data_train_x1) /len(inp_data_train_x1)

def standardize(inp_data_train_x1):
	result = np.empty(len(inp_data_train_x1))
	variances = np.linspace(1,len(inp_data_train_x1),len(inp_data_train_x1))
	mean_x1 = mean_c(inp_data_train_x1)
	for i in range(len(inp_data_train_x1)):
		variances[i] = (inp_data_train_x1[i]-mean_x1)**2
	stdev = sqrt(sum(variances)/(len(inp_data_train_x1)))
	for i in range(len(inp_data_train_x1)):
		result[i] = (inp_data_train_x1[i] - mean_x1)/stdev
	return result, mean_x1, stdev

inp_data_train_x0,mean_train_x0,std_train_x0 = standardize(inp_data_train_x0)
inp_data_train_x1,mean_train_x1,std_train_x1 = standardize(inp_data_train_x1)
inp_data_train_x2,mean_train_x2,std_train_x2 = standardize(inp_data_train_x2)
inp_data_train_x3,mean_train_x3,std_train_x3 = standardize(inp_data_train_x3)
inp_data_train_x4,mean_train_x4,std_train_x4 = standardize(inp_data_train_x4)
inp_data_train = inp_data_train_x0,inp_data_train_x1,inp_data_train_x2,inp_data_train_x3,inp_data_train_x4
inp_data_train = np.transpose(np.asarray(inp_data_train))
#print(inp_data_train)
##################################################################
#fig = plt.figure(figsize = (20,10))
#ax=fig.add_subplot(111,projection="3d")
#plt.title('Standardized Data')
#ax.scatter(inp_data_train_x1, inp_data_train_x2, pred_train, c='r',marker='D')#, cmap='hsv')
#ax.set_xlabel('No of minutes for studying')
#ax.set_ylabel('Ounces of beers taken')
#ax.set_zlabel('Quiz score')
#plt.show()
##################################################################################
def hypothesis_val(w0,w1,w2,w3,w4,w5,x0,x1,x2,x3,x4):
	hx1 = (w0+(w1*x0)+(w2*x1)+(w3*x2)+(w4*x3)+(w5*x4))
	return hx1
def hypothesis(w0,w1,w2,w3,w4,w5,x0,x1,x2,x3,x4):
	hx = np.empty(len(x1))
	for i in range(len(x1)):
		hx[i] = hypothesis_val(w0,w1,w2,w3,w4,w5,x0[i],x1[i],x2[i],x3[i],x4[i])
	return hx
#################################################################
def cost(w0,w1,w2,w3,w4,w5,x0,x1,x2,x3,x4,pred_train):
	tot_count = len(pred_train)
	total_J = 0.0
	Jx = 0.0
	hx = hypothesis(w0,w1,w2,w3,w4,w5,x0,x1,x2,x3,x4)
	#print('inside cost w0',w0,'w1',w1,'w2',w2,'w3',w3,'w4',w4,'w5',w5)

	for i in range(len(pred_train)):
		total_J = total_J+ (hx[i]-pred_train[i])**2
	Jx = (1/(2*tot_count))*total_J
	#print('Jx:',Jx)
	return Jx
#################################################################

def gradient_descent(w0, w1, w2, w3, w4, w5,inp_data_train_x0,inp_data_train_x1,inp_data_train_x2,inp_data_train_x3,inp_data_train_x4, pred_train, alpha, iterations):

	J_history = list()
	iterarr = list()
	w02, w12, w22, w32, w42, w52 = w0, w1, w2, w3, w4, w5
	Jx=0.0
	#print('before iteration start w0',w0,'w1',w1,'w2',w2,'w3',w3,'w4',w4,'w5',w5)
	hx = hypothesis(w0,w1,w2,w3,w4,w5,inp_data_train_x0,inp_data_train_x1,inp_data_train_x2,inp_data_train_x3,inp_data_train_x4)
	Jx = cost(w0,w1,w2,w3,w4,w5,inp_data_train_x0,inp_data_train_x1,inp_data_train_x2,inp_data_train_x3,inp_data_train_x4,pred_train)
	print('Initial cost based on hypothesis',Jx)
	J_history.append(Jx)
	iterarr.append(0)
	for j in range(1,iterations):
		#print('iteration start w0',w0,'w1',w1,'w2',w2,'w3',w3,'w4',w4,'w5',w5)
		J1 = Jx
		for i in range(len(inp_data_train_x1)):
			w0 = w0 - alpha*(1/len(inp_data_train_x1))*np.sum((hx[i]-pred_train[i]))
			w1 = w1 - alpha*(1/len(inp_data_train_x1))*np.sum(((hx[i]-pred_train[i])*(inp_data_train_x0[i])))
			w2 = w2 - alpha*(1/len(inp_data_train_x1))*np.sum(((hx[i]-pred_train[i])*(inp_data_train_x1[i])))
			w3 = w3 - alpha*(1/len(inp_data_train_x1))*np.sum(((hx[i]-pred_train[i])*(inp_data_train_x2[i])))
			w4 = w4 - alpha*(1/len(inp_data_train_x1))*np.sum(((hx[i]-pred_train[i])*(inp_data_train_x3[i])))
			w5 = w5 - alpha*(1/len(inp_data_train_x1))*np.sum(((hx[i]-pred_train[i])*(inp_data_train_x4[i])))
			#print(i)

		#print('\nw0',w0,'\nw1',w1,'\nw2',w2,'\nw3',w3,'\nw4',w4,'\nw5',w5)
		Jx = cost(w0,w1,w2,w3,w4,w5,inp_data_train_x0, inp_data_train_x1,inp_data_train_x2,inp_data_train_x3,inp_data_train_x4,pred_train)
		#print('Cost after alpha reduction (at iteration:',j,') :',Jx)
		J2 = Jx
		#print('\nJ1',J1,'J2',J2)

		if J1<J2:
			break
		J_history.append(Jx)
		iterarr.append(j)

	w02, w12, w22, w32, w42, w52 = w0, w1, w2, w3, w4, w5
	return w02, w12, w22, w32, w42, w52, J_history,iterarr

w0_n,w1_n,w2_n,w3_n,w4_n,w5_n, J_history_n, iterarr_n = gradient_descent(w0, w1, w2, w3, w4, w5,inp_data_train_x0,inp_data_train_x1,inp_data_train_x2, inp_data_train_x3,inp_data_train_x4, pred_train, alpha, iterations)

print('\nafter gradient descent final J value', J_history_n[-1])
plt.scatter(iterarr_n,J_history_n,color='blue',marker='o')
plt.plot(iterarr_n,J_history_n,color='green',marker='D')
plt.xlabel("No of Iterations")
plt.ylabel("J")
plt.show()
######################################################################
def prompt(w0,w1,w2,w3,w4,w5,mean_x0,mean_x1,mean_x2,mean_x3,mean_x4,stdev_x0,stdev_x1,stdev_x2,stdev_x3,stdev_x4):

	inp_data_test_x0 = list()
	inp_data_test_x1 = list()
	inp_data_test_x2 = list()
	inp_data_test_x3 = list()
	inp_data_test_x4 = list()
	while True:
		testval_x0 = float(input('Enter Hour of the day:'))
		testval_x1 = float(input('Enter day of week:'))
		testval_x2 = float(input('Enter floor number considering the lowest as 1st:'))
		testval_x3 = float(input('Enter Temperature:'))
		testval_x4 = float(input('Enter occupancy:'))
		if(testval_x0==0 and testval_x1 == 0 and testval_x2 == 0 and testval_x3 == 0 and testval_x4 == 0):
			break
		else:
#			inp_data_test_x1.append(testval_x)
#			inp_data_test_x2.append(testval_y)
			inp_data_test_x0 = testval_x0
			inp_data_test_x1 = testval_x1
			inp_data_test_x2 = testval_x2
			inp_data_test_x3 = testval_x3
			inp_data_test_x4 = testval_x4
			#########################################################
#
			#print('mean_x1',mean_x1,'mean_x2',mean_x1,'stdev_x1',stdev_x1,'stdev_x2',stdev_x2)
			inp_data_test_x0 = (inp_data_test_x0 - mean_x0)/stdev_x0
			inp_data_test_x1 = (inp_data_test_x1 - mean_x1)/stdev_x1
			inp_data_test_x2 = (inp_data_test_x2 - mean_x2)/stdev_x2
			inp_data_test_x3 = (inp_data_test_x3 - mean_x3)/stdev_x3
			inp_data_test_x4 = (inp_data_test_x4 - mean_x4)/stdev_x4
			#########################################################
			pred_test = hypothesis_val(w0,w1,w2,w3,w4,w5,inp_data_test_x0,inp_data_test_x1,inp_data_test_x2,inp_data_test_x3,inp_data_test_x4)
			#hypothesis(w0,w1,w2,w3,w4,w5,inp_data_test_x1,inp_data_test_x2)
			#print(w0,w1,w2,w3,w4,w5)
			print('Power usage predicted(in watts):',round(pred_test,2))
######################################################################

hx = hypothesis(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,inp_data_train_x0,inp_data_train_x1,inp_data_train_x2,inp_data_train_x3,inp_data_train_x4)
#print('after new hx:',hx,'pred',pred_train,'\nw0',w0_n,'\nw1',w1_n,'\nw2',w2_n,'\nw3',w3_n,'\nw4',w4_n,'\nw5',w5_n)



print('Starting prediction afetr training the model:\n')
inp_data_test_x0_nonstd = inp_data_test_x0
inp_data_test_x1_nonstd = inp_data_test_x1
inp_data_test_x2_nonstd = inp_data_test_x2
inp_data_test_x3_nonstd = inp_data_test_x3
inp_data_test_x4_nonstd = inp_data_test_x4
inp_data_test_x0,mean,std = standardize(inp_data_test_x0)
inp_data_test_x1,mean,std = standardize(inp_data_test_x1)
inp_data_test_x2,mean,std = standardize(inp_data_test_x2)
inp_data_test_x3,mean,std = standardize(inp_data_test_x3)
inp_data_test_x4,mean,std = standardize(inp_data_test_x4)
hx_test = hypothesis(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,inp_data_test_x0,inp_data_test_x1,inp_data_test_x2,inp_data_test_x3,inp_data_test_x4)
#print('Prediction values from test dataset',pred_test)
#iterations = 10000
#w0_nt,w1_nt,w2_nt,w3_nt,w4_nt,w5_nt, J_history_nt, iterarr_nt =gradient_descent(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,inp_data_test_x1,inp_data_test_x2, pred_test, alpha, iterations)
cost_test = cost(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,inp_data_test_x0,inp_data_test_x1,inp_data_test_x2,inp_data_test_x3,inp_data_test_x4,pred_test)
print('\ncost of test data set',cost_test)



#
#fig = plt.figure(figsize = (20,10))
#ax=fig.add_subplot(111,projection="3d")
#plt.title('Test Data')
#ax.scatter(inp_data_test_x1_nonstd, inp_data_test_x2_nonstd, pred_test, c='r',marker='*')#, cmap='hsv')
#ax.set_xlabel('No of minutes for studying')
#ax.set_ylabel('Ounces of beers taken')
#ax.set_zlabel('Quiz score')
#plt.show()


print('\nw0',w0_n,'\nw1',w1_n,'\nw2',w2_n,'\nw3',w3_n,'\nw4',w4_n,'\nw5',w5_n)
prompt(w0_n,w1_n,w2_n,w3_n,w4_n,w5_n,mean_train_x0,mean_train_x1,mean_train_x2,mean_train_x3,mean_train_x4,std_train_x0,std_train_x1,std_train_x2,std_train_x3,std_train_x4)
