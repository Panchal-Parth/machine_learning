#!/usr/bin/python
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

base = int(input('Base: '))
features = int(input('Features: '))
inc = makeCounter_rec(base)
n = [0]* features
print(n)
for i in range(base ** len(n)):
    n = inc(n)
    print(n)

