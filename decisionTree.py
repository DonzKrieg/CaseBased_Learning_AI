import math
from collections import Counter

# Fungsi menghitung entropy dari data
def entropy(data):
    labels = [row[-1] for row in data]  # ambil label
    total = len(labels)
    label_counts = Counter(labels)
    ent = 0.0
    for count in label_counts.values():
        prob = count / total
        ent -= prob * math.log2(prob)
    return ent

# Fungsi membagi data berdasarkan nilai fitur tertentu
def split_data(data, attr_index, value):
    return [row for row in data if row[attr_index] == value]

# Fungsi mencari label mayoritas
def majority_label(data):
    labels = [row[-1] for row in data]
    return Counter(labels).most_common(1)[0][0]

# Fungsi membangun pohon keputusan secara rekursif
def build_tree(data, attributes):
    labels = [row[-1] for row in data]
    
    # Jika semua label sama → return label
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    
    # Jika tidak ada atribut tersisa → return mayoritas
    if not attributes:
        return majority_label(data)
    
    base_entropy = entropy(data)
    best_gain = 0.0
    best_attr = None
    best_splits = None

    for attr_index in attributes:
        values = set([row[attr_index] for row in data])
        new_entropy = 0.0
        splits = {}

        for value in values:
            subset = split_data(data, attr_index, value)
            prob = len(subset) / len(data)
            new_entropy += prob * entropy(subset)
            splits[value] = subset

        info_gain = base_entropy - new_entropy
        if info_gain > best_gain:
            best_gain = info_gain
            best_attr = attr_index
            best_splits = splits

    if best_attr is None:
        return majority_label(data)

    tree = {feature_names[best_attr]: {}}
    remaining_attrs = [i for i in attributes if i != best_attr]

    for value, subset in best_splits.items():
        tree[feature_names[best_attr]][value] = build_tree(subset, remaining_attrs)

    return tree

# Fungsi mencetak pohon keputusan
def print_tree(tree, feature_names, indent=""):
    if isinstance(tree, dict):
        for attr, branches in tree.items():
            for val, subtree in branches.items():
                print(f"{indent}[{attr} = {val}]")
                print_tree(subtree, feature_names, indent + "  ")
    else:
        print(f"{indent}--> {tree}")

# Fungsi prediksi data baru
def predict(tree, row):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    val = row[feature_names.index(attr)]
    if val in tree[attr]:
        return predict(tree[attr][val], row)
    else:
        return "Tidak diketahui"

# ===== DATASET SMARTPHONE =====
feature_names = ['Brand', 'RAM', 'Storage', 'Harga']

dataset = [
    ['Samsung', '6GB', '128GB', 'Mahal', 'Ya'],
    ['Samsung', '4GB', '64GB', 'Murah', 'Tidak'],
    ['Xiaomi', '8GB', '256GB', 'Sedang', 'Ya'],
    ['Xiaomi', '4GB', '64GB', 'Murah', 'Tidak'],
    ['Oppo', '6GB', '128GB', 'Sedang', 'Ya'],
    ['Oppo', '4GB', '64GB', 'Murah', 'Ya'],
    ['iPhone', '8GB', '256GB', 'Mahal', 'Ya'],
    ['iPhone', '4GB', '128GB', 'Mahal', 'Tidak'],
    ['Samsung', '8GB', '256GB', 'Mahal', 'Tidak'],
    ['Xiaomi', '6GB', '128GB', 'Sedang', 'Ya'],
    ['Oppo', '6GB', '64GB', 'Murah', 'Tidak'],
    ['iPhone', '6GB', '128GB', 'Sedang', 'Ya']
]

test_data = [
    ['Samsung', '4GB', '64GB', 'Murah'],
    ['iPhone', '6GB', '128GB', 'Sedang'],
    ['Oppo', '6GB', '64GB', 'Murah'],
    ['Xiaomi', '4GB', '64GB', 'Murah']
]

# ===== PROSES PEMBANGUNAN POHON =====
attributes = list(range(len(feature_names)))  # index fitur
decision_tree = build_tree(dataset, attributes)

# ===== OUTPUT =====
print("==== POHON KEPUTUSAN (Smartphone) ====\n")
print_tree(decision_tree, feature_names)

# ===== PREDIKSI =====
print("\n==== HASIL PREDIKSI ====")
for row in test_data:
    result = predict(decision_tree, row)
    print(f"Data: {row} -> Beli? {result}")
