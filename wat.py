#  def do_things(X, n):
    #  new_list = []
    #  x = [0] * n
    #  for i in range(len(X) ** n):
        #  z = [j - 1 for j in x ]
        #  new_list.append(z)
        #  x[0] += 1
        
        #  for j in range(n):
            #  if x[len(x) - j - 1] >= n:
                #  x[len(x) - j - 1] = 0
                #  x[len(x) - j] += 1
            #  else:
                #  break
    #  return new_list

#  n = int(input('Degree: '))
#  X = [1,2,3,4,5]
#  X_indicies = list(range(len(X)))
#  from itertools import combinations
#  import copy
#  output = sum([list(map(list,combinations(X_indicies,i))) for i in range(len(X_indicies) + 1)],[])
#  o = deepcopy(output)
#  terms = []
#  for item in o:
    #  item = sorted(item)
    #  if len(item) > n or item in terms or len(item) <= 0:
        #  output.remove(item)
    #  else:
        #  terms.append(item)


#  X = [0,1,2,3,4,5]
#  from copy import deepcopy
#  out = []
n = int(input('degree: '))
fc = int(input('features: '))
s = [0] * n
s_index = n - 1
#  while s != [fc] * n:
while s_index >= 0:
    while s[s_index] < fc:
        out.append(deepcopy(s))
        s[s_index] += 1
    s[s_index] = 0
    s_index -= 1
    s[s_index] += 1
    out.append(s)

for l in out:
    print(l)




