import sys

gold = sys.argv[1]

f = open(gold, 'r')

exact = 0
total = 0

for line in sys.stdin:
  g = f.readline()
  if g.strip() == line.strip():
    exact +=1
  total += 1

print(str(exact * 100.0/total))
