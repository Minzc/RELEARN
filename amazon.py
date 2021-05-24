import json
from collections import defaultdict
import pickle
import numpy
from scipy.sparse import csr_matrix

user_embed = {}
item_embed = {}

with open("data/amazon/rate_train_feature.txt") as f:
    for ln in f:
        obj = json.loads(ln)
        x = obj["x"]
        u_id = obj["u_id"]
        i_id = obj["i_id"]
        user_embed[u_id] = x[:64]
        item_embed[i_id] = x[64:]

for i in range(1000):
    assert i in user_embed, i
print(len(user_embed))

item_remap = {}
add_on = len(user_embed)
for i in item_embed:
    item_remap[i] = add_on + i
    assert add_on + i not in user_embed, (add_on, i, user_embed[add_on + i])
    user_embed[item_remap[i]] = item_embed[i]

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
with open("data/amazon/rate_train_feature.txt") as f:
    for ln in f:
        obj = json.loads(ln)
        x = obj["x"]
        u_id = obj["u_id"]
        i_id = obj["i_id"]
        links.append((u_id, item_remap[i_id]))
        graph[u_id].append(item_remap[i_id])
        graph[item_remap[i_id]].append(u_id)

        rows.append(u_id)
        columns.append(item_remap[i_id])
        data.append(1)

with open("data/amazon/link.csv", "w") as w:
    w.write("author_id,author_id\n")
    for u_id, i_id in links:
        w.write(f"{u_id},{i_id}\n")

# =================
# affinity_matrix.p
# =================
affinity_matrix = csr_matrix((data, (numpy.array(rows), numpy.array(columns))), shape=(len(user_embed), len(user_embed)))
pickle.dump(graph, open("data/amazon/affinity_matrix.p", "wb"))

# =================
# graph.p
# =================
pickle.dump(graph, open("data/amazon/graph.p", "wb"))


