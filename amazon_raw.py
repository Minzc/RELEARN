import json
from collections import defaultdict
import pickle
import numpy
from scipy.sparse import csr_matrix

user_embed = {}
item_remap = defaultdict(lambda : 1000 + len(item_remap))
with open("data/amazon/train.txt") as f:
    for ln in f:
        lnsegs = ln.strip().split(" ")
        u_id = int(lnsegs[0])
        user_embed[u_id] = None
        for i in lnsegs[1:]:
            item_remap[int(i)] += 0

for i in range(1000):
    assert i in user_embed, i

item_embed = {}
total_entry = len(user_embed) + len(item_remap)
print(total_entry)
for i in user_embed:
    feature = [0] * total_entry
    feature[i] = 1
    user_embed[i] = feature

for i in item_remap:
    assert item_remap[i] not in user_embed
    feature = [0] * total_entry
    feature[item_remap[i]] = 1
    user_embed[item_remap[i]] = feature

features = []
for i in range(len(user_embed)):
    features.append(user_embed[i])
features = numpy.array(features)
print("features", features.shape)
# =================
# feature.p
# =================
pickle.dump(features, open("data/amazon/features.p", "wb"))

# =================
# link.csv
# =================
links = []
graph = defaultdict(list)
rows = []
columns = []
data = []
with open("data/amazon/train.txt") as f:
    for ln in f:
        lnsegs = ln.strip().split(" ")
        u_id = int(lnsegs[0])
        user_embed[u_id] = None
        for i_id in lnsegs[1:]:
            i_id = int(i_id)
            links.append((u_id, item_remap[i_id]))
            graph[u_id].append(item_remap[i_id])
            graph[item_remap[i_id]].append(u_id)

            rows.append(u_id)
            columns.append(item_remap[i_id])
            assert item_remap[i_id] < total_entry, (item_remap[i_id], total_entry, len(item_remap))
            data.append(1)


with open("data/amazon/link.csv", "w") as w:
    w.write("author_id,author_id\n")
    for u_id, i_id in links:
        w.write(f"{u_id},{i_id}\n")

# =================
# affinity_matrix.p
# =================
affinity_matrix = csr_matrix((data, (numpy.array(rows), numpy.array(columns))),
                             shape=(len(user_embed), len(user_embed)))
pickle.dump(affinity_matrix, open("data/amazon/affinity_matrix.p", "wb"))

# =================
# graph.p
# =================
pickle.dump(graph, open("data/amazon/graph.p", "wb"))
