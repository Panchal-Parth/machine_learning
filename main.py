#! /usr/bin/python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Bennett Meares
CPSC 6820-002
Project 3 - Regression
This project predicts a fish species based on its Body and Dorsal length.
"""
import random
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
all_data = []
all_data_mins = []
all_data_maxes = []
all_data_means = []
test_data = []
train_data = []
dor_data = []
bod_data = []
typ_data = []
W = [159,-3.79,211,-0.0281,-0.0686,-93.7]
temp_W = W
init_J = 0
alpha = 10
init_alpha = alpha
z = [[0], [1], [2], [1,2], [1,1], [2,2]]
iterations = 0
iter_data = []

def main():
    global test_data, train_data, W, iterations, iter_data
    init()
    print('Training, please wait...')
    improve_weights()
    print_results()
    ### display and save plots
    save_scatter3d('scatterplot3d.png')
    save_J_plot('j_plot_full.png', iter_data)

    ### prompt user to enter new data
    while True:
        p = prompt_user()
        if p == (0,0):
            break
        loop(W,p)

"""
name: improve_weights
desc: the primary training driver function
"""
def improve_weights():
    global W, alpha, train_data, temp_W,iterations,iter_data
    init_alpha = alpha
    J_change = -1
    last_J = J(W, train_data) + 1
    current_J = J(W, train_data)
    iter_data.append(current_J) ### include for zero iterations
    while J_change < 0:
        iterations += 1
        for i in range(len(W)):
            temp_W[i] = float(apply_alpha(i))
        print('.', end="", flush=True)
        W = deepcopy(temp_W)
        alpha = float(0.95 * alpha) ### scale alpha by 0.995 to slowly approach best weights
        last_J = current_J
        current_J = J(W, train_data)
        J_change = current_J - last_J
        iter_data.append(current_J) ### store current J value for plotting later

"""
name   : zx_swap
desc   : Used for treating a degree 2 polynomial like a linear model.
         Groups together x's for each weight. For example, z0 corresponds to w0, and z[0] contains
         a list containing 0, which is the index for x0. All x's whose indicies are listed in z
         are multiplied together.
         z is defined above main()
returns: z, otherwise all corresponding x's to a weight with a given index
"""
def zx_swap(index, X):
    global z
    total = float(1)
    for i in z[index]:
        total *= X[i]
    return total

"""
name   : apply_alpha
desc   : Essentially the partial derivative of J. Applies alpha to a weight and the derivative to
         descend J.
returns: New weight value for a weight w
"""
def apply_alpha(index):
    global alpha, train_data, W
    derivative = float(0)
    w = W[index]
    for p in train_data:
        X = (1, p[0],p[1])
        typ = p[2]
        diff = h(W,X) - typ
        derivative += (diff * zx_swap(index, X))
    return w - (alpha * (derivative / float(len(train_data))))

"""
name: sq_err
desc: returns the squared error between a true type and predicted type.
"""
def sq_err(X,typ):
    global W
    pred_typ = h(W,X)
    diff = pred_typ - typ
    return diff ** 2

"""
name   : J
desc   : applies a set of weights (W) against a data set.
returns: cost value of weights W
"""
def J(W, data_set):
    total_sq_error = float(0)
    for p in data_set:
        X = (1, p[0],p[1])
        total_sq_error += float(sq_err(X, p[2]))
    return float(total_sq_error / (2 * len(test_data)))

"""
name   : h
desc   : the hypothesis function. The relation between inputs (X) and weights (W).
returns: predicted typ
"""
def h(W, X):
    y = float(W[0])
    y += W[1] * X[1]
    y += W[2] * X[2]
    y += (W[3] * X[1]) * X[2]
    y += W[4] * (X[1] ** 2)
    y += W[5] * (X[2] ** 2)
    return y

"""
name: init
desc: initializes weights, alpha, and calculates first J(W)
"""
def init():
    global W, temp_W, init_J, test_data, alpha, all_data

    if(len(all_data) == 0):
        filename = input("Enter filename (default: test_data.csv): ")
        if filename == "":
            filename = "test_data.csv"
        read_data(filename)
    divide_data()
    alpha = 1
    init_J = J(W,test_data)

"""
name: read_data
desc: parses a tab-seperated file into several lists
"""
def read_data(filename):
    global all_data, total, bod_data, dor_data, typ_data
    all_data, bod_data, dor_data, typ_data = [],[],[],[]
    with open(filename,"r") as file:
        for line in file:
            d = line.strip().split(',')
            if len(d) == 1:
                total = int(d[0])
                continue
            bod = int(d[0])
            dor = int(d[1])
            typ = float(d[2])
            all_data.append((bod,dor,typ))
            bod_data.append(bod)
            dor_data.append(dor)
            typ_data.append(typ)
            #  if typ == 0:
                #  tiger0.data.append((bod,dor,typ))
                #  tiger0.bodies.append(bod)
                #  tiger0.dorsals.append(dor)
            #  elif typ == 1:
                #  tiger1.data.append((bod,dor,typ))
                #  tiger1.bodies.append(bod)
                #  tiger1.dorsals.append(dor)
            #  else:
                #  print('Incorrect type. Ignoring...')
                #  continue


"""
name: divide_data
desc: shuffles and splits data into training and test sets
"""
def divide_data():
    global train_data, test_data, all_data
    all_data_stats()
    all_data = random.sample(all_data, len(all_data))
    all_data = scale_data_set(all_data)
    tr_index = int(0.8 * len(all_data))
    train_data = all_data[:tr_index]
    test_data = all_data[tr_index:]


"""
name    : all_data_stats
desc    : calulates minimums, maximums, and means of features in all_data
"""
def all_data_stats():
    global all_data, all_data_mins, all_data_maxes, all_data_means
    x1_total,x2_total,x1_max,x2_max,x1_min,x2_min = 0,0,0,0,0,0
    for p in all_data:
        x1_total += p[0]
        x2_total += p[1]
        if(p[0] > x1_max):
            x1_max = p[0]
        if(p[1] > x2_max):
            x2_max = p[1]
        if(p[0] < x1_min):
            x1_min = p[0]
        if(p[1] < x2_min):
            x2_min = p[1]

    x1_mean = float(x1_total / len(all_data))
    x2_mean = float(x2_total / len(all_data))
    
    all_data_maxes = (x1_max,x2_max)
    all_data_mins = (x1_min,x2_min)
    all_data_means = (x1_mean,x2_mean)


"""
name    : scale_data
desc    : scales every point in a data set
returns : list of tuples (new data set)
"""
def scale_data_set(data):
    global all_data_mins, all_data_maxes, all_data_means
    scaled_data = []

    for p in data:
        scaled_data.append(scale_point(p,all_data_maxes,all_data_mins,all_data_means))
    return scaled_data


"""
name    : scale_point
desc    : scales features in a point according to mean normalization
returns : tuple (bod,dors,typ)
"""
def scale_point(p,maxes,mins,means):
    x1 = p[0]
    x2 = p[1]
    x1_max = maxes[0]
    x2_max = maxes[1]
    x1_min = mins[0]
    x2_min = mins[1]
    x1_mean = means[0]
    x2_mean = means[1]

    ##  scale the data
    x1 = (x1 - x1_mean) / (x1_max - x1_min)
    x2 = (x2 - x2_mean) / (x2_max - x2_min)
    return (x1,x2,p[2])


"""
name    : un_scale_data
desc    : un_scales every point in a data set
returns : list of tuples (new data set)
"""
def un_scale_data(data):
    global all_data_mins, all_data_maxes, all_data_means
    un_scaled_data = []
    for p in data:
        un_scaled_data.append(un_scale_point(p,all_data_maxes,all_data_mins,all_data_means))
    return un_scaled_data
    

"""
name    : un_scale_point
desc    : converts a scaled point to its original magnitude
returns : tuple (bod,dors,typ)
"""
def un_scale_point(s_p, maxes, mins, means):
    sx1 = s_p[0]
    sx2 = s_p[1]
    x1_max = maxes[0]
    x2_max = maxes[1]
    x1_min = mins[0]
    x2_min = mins[1]
    x1_mean = means[0]
    x2_mean = means[1]

    ##  scale the data
    x1 = (sx1 * (x1_max - x1_min)) + x1_mean
    x2 = (sx2 * (x2_max - x2_min)) + x2_mean
    return (x1,x2,s_p[2])


"""
name   : prompt_user
desc   : asks the user for values
returns: tuple of dorsal and body length
"""
def prompt_user():
    global all_data_maxes,all_data_mins,all_data_means
    error = "Please enter the correct value."
    while True:
        try:
            bod = int(get_date())
        except:
            print(error)
        else:
            break
    while True:
        try:
            dor = float(input("Building occupancy (int) : "))
        except :
            print(error)
        else:
            break
    if(bod == 0 and dor == 0):
        return (0,0)
    p = (bod,dor,-1)
    return scale_point(p, all_data_maxes,all_data_mins,all_data_means)

#  def accuracy(tp, tn, fp, fn):
    #  return (tp + tn) / (float(tp + tn + fp + fn))

#  def precision(tp, fp):
    #  return tp / (float(tp + fp))

#  def recall(tp, fn):
    #  return tp / (float(tp + fn))

#  def f1(prec, rec):
    #  denom = (1 / float(prec)) + (1 / float(rec))
    #  return 2 * (1 / denom)


"""
name: print_results
desc: prints relevant information after training, such as number of iterations, alpha values,
      J(W) values, and the model definition
"""
def print_results():
    global iterations, alpha, init_alpha, W, test_data, init_J
    #  c = confusion_matrix()
    #  tn = c[0]
    #  fp = c[1]
    #  fn = c[2]
    #  tp = c[3]
    #  acc = accuracy(tp, tn, fp, fn)
    #  prec = precision(tp, fp)
    #  rec = recall(tp, fn)
    #  f = f1(prec, rec)
    
    print("\nRESULTS:\n----------")
    print("Training iterations   :", iterations)
    print("Initial alpha value   :", init_alpha)
    print("End alpha value       : {:.2e}".format(alpha))
    print("Initial J(W)(test)    : {:.2e}".format(init_J))
    print("Final J(W)(test)      : {:.2e}".format(J(W,test_data)))
    print("Hypothesis model      :", equation_string(W))
    #  print('\nCONFUSION MATRIX:')
    #  print('----------')
    #  print("TN:{:3d}".format(tn), " | FP:{:3d}".format(fp))
    #  print("FN:{:3d}".format(fn), " | TP:{:3d}".format(tp))
    #  print('\naccuracy              :', round(acc,4))
    #  print('precision             :', round(prec,4))
    #  print('recall                :', round(rec,4))
    #  print('f1                    :', round(f,4))


"""
name   : equation_string
desc   : formats the weights into a human-readable string
returns: the mathematical definition of the hypothesis function (h(X)) as a string
"""
def equation_string(W):
    out = "h(X) = "
    out += "({:.2e}".format(W[0]) + ") + " + "({:.2e})(x1) + ".format(W[1])
    out += "({:.2e})(x2) + ".format(W[2]) + "({:.2e})(x1)(x2) + ".format(W[3])
    out += "({:.2e})(x1 ^ 2) + ".format(W[4]) + "({:.2e})(x2 ^ 2)".format(W[5])
    return out


#  """
#  name    : condidence
#  desc    : converts the scalar pred_type to a percentage
          #  ex: pred_type of 0.25 is a confidence rate of 50% for tigerFish0
          #  because 0.5 is halfway between 0 and 0.5 (cutoff for classification)
#  returns : percentage string
#  """
#  def confidence(pred_typ):
    #  if(pred_typ > 0.5):
        #  rel_per = (pred_typ - 0.5) / 0.5
    #  else:
        #  rel_per = 1 - (pred_typ / 0.5)

    #  per = rel_per * 100
    #  if(per >= 100):
        #  per = 99.99
    #  string = str(round(per,2)) + '%'

    #  return string

#  """
#  name    : confusion_matric
#  desc    : calculates true and false negative/positives
#  """
#  def confusion_matrix():
    #  tn,fp,fn,tp = 0,0,0,0
    #  for p in test_data:
        #  X = (1, p[0], p[1])
        #  pred_typ = (h(W,X))
        #  if pred_typ < 0.5:
            #  pred_typ = 0
        #  else:
            #  pred_typ = 1

        #  pred_actual_tuple = (pred_typ, p[2])
        #  if pred_actual_tuple[0] == 0 and pred_actual_tuple[1] == 0:
            #  tp += 1
        #  elif pred_actual_tuple[0] == 0 and pred_actual_tuple[1] == 1:
            #  fn += 1
        #  elif pred_actual_tuple[0] == 1 and pred_actual_tuple[1] == 0:
            #  fp += 1
        #  elif pred_actual_tuple[0] == 1 and pred_actual_tuple[1] == 1:
            #  tn += 1
        #  else:
            #  print(pred_actual_tuple)
    #  return tn,fp,fn,tp


"""
name: loop
desc: driver script for calculating and printing predicted types
"""
def loop(W,p):
    global tiger0,tiger1
    X = (1, p[0], p[1])
    pred_typ = (h(W,X))
    print("Predicted Usage        :", pred_typ)

"""
name: save_scatter3d
desc: displays and writes 3D scatterplot
"""
def save_scatter3d(filename):
    global bod_data, dor_data, all_data
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('Building Occupancy (int)')
    ax.set_ylabel('Unix Time (int)')
    ax.set_zlabel('Power Usage (kW)')
    plt.title('Time vs Occupancy in Power Use')
    ax.scatter(dor_data, bod_data, typ_data, marker="v", label="")
    plt.savefig(filename)
    plt.show()
    plt.clf()

"""
name: save_J_plot
desc: displays and saves a plot of J-values with any given begin and end indicies
"""
def save_J_plot(filename, data, begin = 0, end = None):
    if end is None:
        end = len(data)
    sub_data = data[begin:end]
    plt.xticks(np.arange(begin,end,int(0.1 * (end - begin))))
    t = np.arange(begin,end,1)
    plt.xlabel('Iterations (' + str(begin) + '—' + str(end) + ')')
    plt.ylabel('J(W)')
    plt.title('Model Accuracy After ' + str((end)) + ' Iterations')
    figure_text = 'J(W) values of note:\nMin: ' + "{:.2e}".format(min(sub_data)) + '\n  (' + str(data.index(min(sub_data))) + ' iterations)'
    figure_text += '\nMax: ' + "{:.2e}".format(max(sub_data)) + '\n  (' + str(data.index(max(sub_data))) + ' iterations)'
    plt.figtext(0.5,0.65,figure_text) 
    plt.plot(t, data[begin:end], marker="o")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    plt.clf()

import datetime
def get_date():
    error = "\nEnter the date in this exact format: yyyy-mm-dd hr:min\n"
    while True:
        try:
            first,second = "",""
            result = input('\nEnter date (yyyy-mm-dd hr:min): ').split(' ')
            first,second = result
            yr = int(first.split('-')[0])
            mo = int(first.split('-')[1])
            dy = int(first.split('-')[2])
            hr = int(second.split(':')[0])
            mn = int(second.split(':')[1])

        except:
            if result == ['0']:
                return 0
            print(error)
        else:
            break
    return date_to_int(yr,mo,dy,hr,mn)

def date_to_int(yr,mo,dy,hr = 0, mn = 0):
    
    u = int(datetime.datetime(yr,mo,dy,hr,mn).timestamp())
    return u


if __name__ == "__main__":
    main()
