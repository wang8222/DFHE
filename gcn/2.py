import re

with open('edge.txt') as f:
  data = f.read()

data = re.sub('u-co', 'u-o', data)
data = re.sub('co-u', 'o-u', data)
data = re.sub('b-ca', 'b-a', data)
data = re.sub('ca-b', 'a-b', data)
data = re.sub('b-ci', 'b-i', data)
data = re.sub('ci-b', 'i-b', data)
with open('edge.txt', 'w') as f:
  f.write(data)