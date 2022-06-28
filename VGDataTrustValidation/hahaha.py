from collections import Counter
import sys


a = []
[a.append('name') for i in range(10000)]
[a.append('jsm') for i in range(200)]
[a.append('banana') for i in range(500)]
[a.append('minions') for i in range(700)]

cnt = Counter(a)
print(cnt['name'])
print(cnt['jsm'])
print(cnt['banana'])
print(cnt['minions'])
