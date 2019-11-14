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
from itertools import combinations
all_data = []
all_data_mins = []
all_data_maxes = []
all_data_means = []
test_data = []
train_data = []
dor_data = []
bod_data = []
typ_data = []
W = []
feature_count = 0
feature_labels = []
temp_W = W
init_J = 0
alpha = 10
init_alpha = alpha
z = []
degree = 2
iterations = 0
iter_data = []

def main():
    global test_data, train_data, W, iterations, iter_data
    init()
    print('Training, please wait...')
    improve_weights()
    print_results()
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
        alpha = float(0.995 * alpha) ### scale alpha by 0.995 to slowly approach best weights
        last_J = current_J
        current_J = J(W, train_data)
        J_change = current_J - last_J
        iter_data.append(current_J) ### store current J value for plotting later


def build_z(d):
    global z, feature_count
    def recur(n):
        if n <= 0:
            yield []
        else:
            for result in recur(n - 1):
                for i in range(d + 1):
                    x = _X[i]
                    terms.append(result)
                    yield [*result,x]

    def makeCounter_rec(base):
        def incDigit(num, pos):
            new = num[:]
            if(num[pos] == base - 1):
                new[pos] = 0
                if(pos < len(num) - 1):
                    return incDigit(new, pos + 1)
            else:
                new[pos] = new[pos] + 1
            return new

        def inc(num):
            return incDigit(num, 0)
        return inc

    terms = []
    base = int(feature_count + 1)
    inc = makeCounter_rec(base)
    combo = [0] * int(d)
    z.append(combo)
    terms.append(combo)
    for i in range(base ** len(combo)):
        combo = inc(combo)
        z_ = sorted(deepcopy(combo))
        if z_ not in terms:
            terms.append(z_)
            z.append(z_)

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
        x_list = [1]
        for f in p:
            x_list.append(f)
        X = tuple(x_list)
        typ = p[-1]
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
        X = [1]
        for f in p[:-1]:
            X.append(f)
        total_sq_error += float(sq_err(X, p[-1]))
    return float(total_sq_error / (2 * len(test_data)))

"""
name   : h
desc   : the hypothesis function. The relation between inputs (X) and weights (W).
returns: predicted typ
"""
def h_old(W, X):
    y = float(W[0])
    y += W[1] * X[1]
    y += W[2] * X[2]
    y += (W[3] * X[1]) * X[2]
    y += W[4] * (X[1] ** 2)
    y += W[5] * (X[2] ** 2)
    return y

def h(W, X):
    global degree
    y = 0.0
    for i in range(degree + 1):
        y += W[i] * zx_swap(i,X)
    return y


"""
name: init
desc: initializes weights, alpha, and calculates first J(W)
"""
def init():
    global W, temp_W, init_J, test_data, alpha, all_data, degree, z, feature_count

    default = 2
    def_filename = "label_new_data_set.csv"
    degree = input("What degree polynomial would you like to model? (default: " + str(default) + "): ")
    if degree == "":
        degree = int(default)
    else:
        degree = int(degree)
    if(len(all_data) == 0):
        filename = input("Enter filename (default: " + def_filename + "): ")
        if filename == "":
            filename = def_filename
        read_data(filename)
    build_z(degree)
    W = [0.0] * len(z)
    temp_W = W
    print(z)
    input()

    divide_data()
    alpha = 1
    init_J = J(W,test_data)

"""
name: read_data
desc: parses a tab-seperated file into several lists
"""
def read_data(filename):
    global all_data, total, bod_data, dor_data, typ_data, feature_count, feature_labels
    all_data, bod_data, dor_data, typ_data = [],[],[],[]
    with open(filename,"r") as file:
        for line in file:
            d = line.strip().split(',')
            if len(d) == 1:
                total = int(d[0])
                continue
            try:
                d = [float(i) for i in d]
            except:
                feature_labels = d
                feature_count = len(d) - 1
                continue
            all_data.append(tuple(d))

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
    global all_data, all_data_mins, all_data_maxes, all_data_means, feature_count
    x_i_total,x_i_maxes,x_i_mins = [0.0] * feature_count,[0.0] * feature_count,[0.0] * feature_count
    x_i_means = [0.0] * feature_count
    for p in all_data:
        for i in range(len(p) - 1):
            x_i_total[i] += p[i]
            if p[i] > x_i_maxes[i]:
                x_i_maxes[i] = p[i]
            if p[i] < x_i_mins[i]:
                x_i_mins[i] = p[i]

    for f in range(feature_count):
        x_i_means[i] = float(x_i_total[i] / len(all_data))
    
    all_data_maxes = tuple(x_i_maxes)
    all_data_mins = tuple(x_i_mins)
    all_data_means = tuple(x_i_means)


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
    y = p[-1]
    new_point_list = []

    for i in range(len(maxes)):
        x_i = p[i]
        x_i_max = maxes[i]
        x_i_min = mins[i]
        x_i_mean = means[i]
        x_i = (x_i - x_i_mean) / (x_i_max - x_i_min)
        new_point_list.append(x_i)
    new_point_list.append(y)

    return tuple(new_point_list)


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
    y = p[:-1]
    new_point_list = []

    for i in range(len(maxes)):
        x_i = p[i]
        x_i_max = maxes[i]
        x_i_min = mins[i]
        x_i_mean = means[i]
        x_i = (x_i * (x_i_max - x_i_min)) + x_i_mean
        new_point_list.append(x_i)
    new_point_list.append(y)

    return tuple(new_point_list)


"""
name   : prompt_user
desc   : asks the user for values
returns: tuple of dorsal and body length
"""
def prompt_user():
    global all_data_maxes,all_data_mins,all_data_means,feature_count
    error = "Please enter the correct value."

    feats = [0] * int(feature_count)
    for i in range(feature_count):
        while True:
            try:
                l = feature_labels[i]
                prompt = str(l) + ": "
                f = float(input(prompt))
                
            except:
                print(error)
                break
            else:
                break
        feats.append(f)

    #  if(feats == [0.0] * feature_count):
        #  return tuple([0.0] * feature_count)
    feats.append(-1)
    p = tuple(feats)
    return scale_point(p, all_data_maxes,all_data_mins,all_data_means)

"""
name: print_results
desc: prints relevant information after training, such as number of iterations, alpha values,
      J(W) values, and the model definition
"""
def print_results():
    global iterations, alpha, init_alpha, W, test_data, init_J
   
    print("\nRESULTS:\n----------")
    print("Training iterations   :", iterations)
    print("Initial alpha value   :", init_alpha)
    print("End alpha value       : {:.2e}".format(alpha))
    print("Initial J(W)(test)    : {:.2e}".format(init_J))
    print("Final J(W)(test)      : {:.2e}".format(J(W,test_data)))
    print("Hypothesis model      :", equation_string(W))

"""
name   : equation_string
desc   : formats the weights into a human-readable string
returns: the mathematical definition of the hypothesis function (h(X)) as a string
"""
def equation_string(W):
    global z
    out = "h(X) = "
    w_i = 0
    for x_i_list in z:
        out += "({:.2e})".format(W[w_i])
        for x_i in x_i_list:
            if x_i > 0:
                out += "(x" + str(x_i) + ")"
        out += " + "
        w_i += 1

    return out[:-3]


"""
name: loop
desc: driver script for calculating and printing predicted types
"""
def loop(W,p):
    global tiger0,tiger1
    x_list = [1]
    for f in p:
        x_list.append(f)
    X = tuple(x_list)
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
