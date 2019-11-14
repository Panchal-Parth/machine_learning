#! /usr/bin/python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

X = ["1","x1","x2","x3","x4","x5","x6"]
X = ["0","1","2","3","4","5"]
z = []

out = ""
terms = []
_X = []

queue = []
#  degree = len(X) - 1
degree = 2

test = []
test = [0.0] * 5
print(test)



def recur(n):
    if n <= 0:
        yield []
    else:
        for result in recur(n - 1):
            #  if result not in terms:
            for i in range(degree + 1):
                x = X[i]
                terms.append(result)
                yield [*result,x]

l = recur(degree)
for h in l:
    h = sorted(h)
    if h not in terms:
        terms.append(h)
        print(h)
        s = ""
        for x in h:
            s += x + "*"
        s = s[:-1]
        _X.append(s)
        z.append(h)

print(z)
#  for x in X:
    #  for _x in X:
        #  for __x in X:
            #  for ___x in X:
                #  t = sorted([x,_x,__x,___x])
                #  if t not in terms:
                    #  _X.append(str(x + "*" + _x + "*" + __x + "*" + ___x))
                    #  terms.append(t)

for x in _X:
    out += x + " + "
out = out[:-3]


print(out)


