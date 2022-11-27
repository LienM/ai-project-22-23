from annoy import AnnoyIndex
import random
import time

start = time.time()
f = 40  # Length of item vector that will be indexed

t = AnnoyIndex(f, 'angular')
for i in range(1_000_000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10)  # 10 trees
t.save('test.ann')

u = AnnoyIndex(f, 'angular')
u.load('test.ann')  # superfast, will just mmap the file
v = [random.gauss(0, 1) for z in range(f)]
print(u.get_nns_by_vector(v, 20))  # will find the 1000 nearest neighbors
print(time.time() - start, "seconds")
