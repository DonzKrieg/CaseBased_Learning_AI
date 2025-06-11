# Mengimpor modul yang diperlukan
import math  # Diperlukan untuk fungsi logaritma (log2) dalam perhitungan entropy.
from collections import Counter  # Diperlukan untuk menghitung frekuensi setiap label dengan mudah.

# --- FUNGSI-FUNGSI PEMBANTU ---

def entropy(data):
    """Fungsi ini menghitung nilai entropy dari sebuah dataset."""
    
    # 1. Ekstrak semua label (kolom terakhir) dari setiap baris data.
    labels = [row[-1] for row in data]  
    
    # 2. Jika tidak ada data/label, maka entropy-nya 0 (tidak ada ketidakpastian).
    if not labels:
        return 0
        
    # 3. Hitung jumlah total data.
    total = len(labels)
    
    # 4. Hitung berapa kali setiap label unik muncul (misal: {'Ya': 8, 'Tidak': 4}).
    label_counts = Counter(labels)
    
    # 5. Inisialisasi variabel entropy dengan nilai awal 0.0.
    ent = 0.0
    
    # 6. Looping untuk setiap jumlah hitungan label (misal: 8 dan 4).
    for count in label_counts.values():
        # 6a. Hitung probabilitas kemunculan label tersebut.
        prob = count / total
        # 6b. Kurangi nilai entropy dengan (probabilitas * log2(probabilitas)).
        ent -= prob * math.log2(prob)
        
    # 7. Kembalikan nilai akhir entropy.
    return ent

def split_data(data, attr_index, value):
    """Fungsi ini membagi dataset menjadi sub-dataset berdasarkan nilai atribut tertentu."""
    
    # Mengembalikan list baru yang hanya berisi baris-baris di mana nilai pada kolom 'attr_index' sama dengan 'value'.
    return [row for row in data if row[attr_index] == value]

def majority_label(data):
    """Fungsi ini menemukan label (kelas) yang paling sering muncul dalam sebuah dataset."""
    
    # 1. Ekstrak semua label dari data.
    labels = [row[-1] for row in data]
    
    # 2. Jika tidak ada label, kembalikan None (tidak ada mayoritas).
    if not labels:
        return None 
        
    # 3. Gunakan Counter.most_common(1) untuk mendapatkan pasangan (label, jumlah) yang paling umum, lalu ambil labelnya.
    return Counter(labels).most_common(1)[0][0]

# --- FUNGSI UTAMA PEMBANGUNAN POHON ---

def build_tree(data, attributes):
    """Fungsi ini membangun pohon keputusan (Decision Tree) secara rekursif."""
    
    # 1. Ekstrak semua label dari data saat ini.
    labels = [row[-1] for row in data]
    
    # === KASUS DASAR (KONDISI BERHENTI) ===
    
    # 2. Jika semua data memiliki label yang sama, buat simpul daun dengan label tersebut.
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    
    # 3. Hitung label mayoritas untuk simpul saat ini. Ini akan digunakan sebagai prediksi default jika nanti ada data tak terlihat.
    default = majority_label(data)

    # 4. Jika tidak ada atribut tersisa untuk diuji, buat simpul daun dengan label mayoritas.
    if not attributes:
        return default
    
    # === PROSES REKURSIF ===
    
    # 5. Hitung entropy dari dataset saat ini (sebelum dipecah).
    base_entropy = entropy(data)
    
    # 6. Inisialisasi variabel untuk menyimpan information gain, atribut, dan cara pemecahan data terbaik.
    best_gain = 0.0
    best_attr = None
    best_splits = {}

    # 7. Loop melalui setiap indeks atribut yang tersisa.
    for attr_index in attributes:
        # 7a. Dapatkan semua nilai unik dari atribut yang sedang diuji.
        values = set([row[attr_index] for row in data])
        # 7b. Inisialisasi entropy baru (setelah dipecah) dengan 0.
        new_entropy = 0.0
        # 7c. Siapkan dictionary untuk menyimpan sub-dataset hasil pemecahan.
        splits_for_attr = {}
        # 7d. Loop melalui setiap nilai unik.
        for value in values:
            # Pecah data berdasarkan nilai ini.
            subset = split_data(data, attr_index, value)
            # Hitung probabilitas dari sub-dataset ini.
            prob = len(subset) / len(data)
            # Tambahkan entropy tertimbang dari sub-dataset ke entropy baru.
            new_entropy += prob * entropy(subset)
            # Simpan sub-dataset hasil pemecahan.
            splits_for_attr[value] = subset
        
        # 7e. Hitung information gain untuk atribut ini.
        info_gain = base_entropy - new_entropy
        
        # 7f. Jika gain saat ini lebih baik dari yang terbaik sejauh ini, perbarui variabel terbaik.
        if info_gain > best_gain:
            best_gain = info_gain
            best_attr = attr_index
            best_splits = splits_for_attr

    # 8. Jika tidak ada atribut yang memberikan gain (semua gain=0), buat simpul daun dengan label mayoritas.
    if best_gain == 0:
        return default

    # 9. Buat simpul pohon baru (berupa dictionary) dengan nama atribut terbaik sebagai kunci utama.
    tree = {feature_names[best_attr]: {}}

    # 10. Simpan prediksi default (label mayoritas) di simpul ini menggunakan kunci khusus '_default'.
    tree[feature_names[best_attr]]['_default'] = default
    
    # 11. Siapkan daftar atribut yang tersisa untuk pemanggilan rekursif berikutnya.
    remaining_attrs = [i for i in attributes if i != best_attr]

    # 12. Loop melalui setiap sub-dataset hasil pemecahan terbaik.
    for value, subset in best_splits.items():
        # 12a. Bangun subtree secara rekursif dan simpan hasilnya di dalam pohon.
        tree[feature_names[best_attr]][value] = build_tree(subset, remaining_attrs)

    # 13. Kembalikan pohon yang sudah jadi.
    return tree

# --- FUNGSI-FUNGSI UNTUK OUTPUT DAN PREDIKSI ---

def print_tree(tree, indent=""):
    """Fungsi ini mencetak struktur pohon keputusan dengan rapi menggunakan indentasi."""
    
    # Jika 'tree' bukan dictionary, berarti ini adalah simpul daun (hasil akhir). Cetak nilainya.
    if not isinstance(tree, dict):
        print(f"{indent}--> {tree}")
        return

    # Ambil nama atribut dari kunci utama dictionary.
    attribute = next(iter(tree))
    # Ambil semua cabang dari atribut tersebut.
    branches = tree[attribute]
    
    # Loop melalui setiap nilai dan subtree di dalam cabang.
    for value, subtree in branches.items():
        # Abaikan kunci '_default' agar tidak ikut tercetak di struktur pohon.
        if value == '_default':
            continue
        # Cetak kondisi (misal: [Harga = Murah]).
        print(f"{indent}[{attribute} = {value}]")
        # Panggil fungsi print_tree secara rekursif untuk subtree, dengan indentasi yang lebih dalam.
        print_tree(subtree, indent + "  ")

def predict(tree, row):
    """Fungsi ini memprediksi label untuk satu baris data menggunakan pohon keputusan yang sudah ada."""
    
    # Kasus Dasar: Jika 'tree' bukan dictionary, berarti kita sudah sampai di simpul daun. Kembalikan nilainya.
    if not isinstance(tree, dict):
        return tree

    # Ambil nama atribut dari simpul saat ini.
    attribute = next(iter(tree))
    # Ambil semua cabang dari simpul saat ini.
    branches = tree[attribute]
    # Ambil nilai dari data tes yang sesuai dengan atribut simpul saat ini.
    value = row[feature_names.index(attribute)]

    # Jika nilai dari data tes ada di dalam cabang pohon...
    if value in branches:
        # ...lanjutkan prediksi secara rekursif ke subtree yang sesuai.
        return predict(branches[value], row)
    else:
        # Jika tidak ada (data tidak terlihat saat training), kembalikan prediksi default yang tersimpan di simpul ini.
        return branches['_default']

# --- DATA DAN EKSEKUSI UTAMA ---

# Mendefinisikan nama-nama fitur/kolom secara berurutan.
feature_names = ['Brand', 'RAM', 'Storage', 'Harga']

# Dataset untuk melatih model Decision Tree.
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
    ['iPhone', '6GB', '128GB', 'Sedang', 'Ya'],
    ['Samsung', '6GB', '64GB', 'Sedang', 'Ya'],
    ['Xiaomi', '6GB', '64GB', 'Murah', 'Tidak'],
    ['Oppo', '8GB', '256GB', 'Mahal', 'Ya'],
    ['iPhone', '6GB', '64GB', 'Sedang', 'Tidak'],
    ['Samsung', '4GB', '128GB', 'Murah', 'Tidak'],
    ['Xiaomi', '4GB', '128GB', 'Sedang', 'Ya'],
    ['Oppo', '4GB', '128GB', 'Sedang', 'Tidak'],
    ['iPhone', '8GB', '128GB', 'Mahal', 'Ya'],
    ['Samsung', '8GB', '128GB', 'Mahal', 'Ya'],
    ['Xiaomi', '8GB', '128GB', 'Sedang', 'Ya'],
    ['Oppo', '8GB', '128GB', 'Mahal', 'Ya'],
    ['iPhone', '4GB', '64GB', 'Murah', 'Tidak'],
    ['Xiaomi', '6GB', '256GB', 'Sedang', 'Ya'],
    ['Samsung', '6GB', '256GB', 'Sedang', 'Ya'],
    ['Oppo', '6GB', '256GB', 'Mahal', 'Ya'],
    ['iPhone', '6GB', '256GB', 'Mahal', 'Ya'],
    ['Samsung', '4GB', '256GB', 'Sedang', 'Tidak'],
    ['Xiaomi', '4GB', '256GB', 'Murah', 'Tidak'],
    ['Oppo', '4GB', '256GB', 'Sedang', 'Tidak'],
    ['iPhone', '4GB', '256GB', 'Sedang', 'Tidak'],
    ['Samsung', '8GB', '64GB', 'Mahal', 'Tidak'],
    ['Xiaomi', '8GB', '64GB', 'Sedang', 'Ya'],
    ['Oppo', '8GB', '64GB', 'Mahal', 'Ya'],
    ['iPhone', '8GB', '64GB', 'Mahal', 'Ya'],
    ['Samsung', '6GB', '128GB', 'Sedang', 'Ya'],
    ['Xiaomi', '6GB', '128GB', 'Murah', 'Ya'],
    ['Oppo', '6GB', '128GB', 'Murah', 'Tidak'],
    ['iPhone', '6GB', '128GB', 'Mahal', 'Ya'],
    ['Samsung', '4GB', '64GB', 'Murah', 'Tidak'],
    ['Xiaomi', '4GB', '64GB', 'Murah', 'Tidak'],
    ['Oppo', '4GB', '64GB', 'Murah', 'Ya'],
    ['iPhone', '4GB', '64GB', 'Murah', 'Tidak'],
    ['Samsung', '8GB', '256GB', 'Mahal', 'Ya'],
    ['Xiaomi', '8GB', '256GB', 'Sedang', 'Ya'],
    ['Oppo', '8GB', '256GB', 'Mahal', 'Ya'],
    ['iPhone', '8GB', '256GB', 'Mahal', 'Ya'],
    ['Samsung', '6GB', '64GB', 'Sedang', 'Ya'],
    ['Xiaomi', '6GB', '64GB', 'Murah', 'Tidak'],
    ['Oppo', '6GB', '64GB', 'Sedang', 'Tidak'],
    ['iPhone', '6GB', '64GB', 'Sedang', 'Tidak'],
    ['Samsung', '4GB', '128GB', 'Murah', 'Tidak'],
    ['Xiaomi', '4GB', '128GB', 'Sedang', 'Ya'],
    ['Oppo', '4GB', '128GB', 'Sedang', 'Tidak'],
    ['iPhone', '4GB', '128GB', 'Sedang', 'Tidak'],
    ['Samsung', '8GB', '128GB', 'Mahal', 'Ya'],
    ['Xiaomi', '8GB', '128GB', 'Sedang', 'Ya'],
    ['Oppo', '8GB', '128GB', 'Mahal', 'Ya'],
    ['iPhone', '8GB', '128GB', 'Mahal', 'Ya'],
    ['Samsung', '6GB', '256GB', 'Mahal', 'Ya'],
    ['Xiaomi', '6GB', '256GB', 'Sedang', 'Ya'],
    ['Oppo', '6GB', '256GB', 'Sedang', 'Ya'],
    ['iPhone', '6GB', '256GB', 'Mahal', 'Ya'],
    ['Samsung', '4GB', '256GB', 'Murah', 'Tidak'],
    ['Xiaomi', '4GB', '256GB', 'Murah', 'Tidak'],
    ['Oppo', '4GB', '256GB', 'Sedang', 'Tidak'],
    ['iPhone', '4GB', '256GB', 'Murah', 'Tidak'],
    ['Samsung', '8GB', '64GB', 'Mahal', 'Ya'],
    ['Xiaomi', '8GB', '64GB', 'Sedang', 'Ya'],
    ['Oppo', '8GB', '64GB', 'Mahal', 'Ya'],
    ['iPhone', '8GB', '64GB', 'Mahal', 'Ya'],
    ['Samsung', '6GB', '128GB', 'Sedang', 'Ya'],
    ['Xiaomi', '6GB', '128GB', 'Murah', 'Ya'],
    ['Oppo', '6GB', '128GB', 'Sedang', 'Tidak'],
    ['iPhone', '6GB', '128GB', 'Mahal', 'Ya'],
    ['Samsung', '4GB', '64GB', 'Murah', 'Tidak'],
    ['Xiaomi', '4GB', '64GB', 'Murah', 'Tidak'],
    ['Oppo', '4GB', '64GB', 'Murah', 'Ya'],
    ['iPhone', '4GB', '64GB', 'Murah', 'Tidak'],
    ['Samsung', '8GB', '256GB', 'Mahal', 'Ya'],
    ['Xiaomi', '8GB', '256GB', 'Sedang', 'Ya'],
    ['Oppo', '8GB', '256GB', 'Mahal', 'Ya'],
    ['iPhone', '8GB', '256GB', 'Mahal', 'Ya'],
    ['Samsung', '6GB', '64GB', 'Sedang', 'Ya'],
    ['Xiaomi', '6GB', '64GB', 'Murah', 'Tidak'],
    ['Oppo', '6GB', '64GB', 'Sedang', 'Tidak'],
    ['iPhone', '6GB', '64GB', 'Sedang', 'Tidak'],
    ['Samsung', '4GB', '128GB', 'Murah', 'Tidak'],
    ['Xiaomi', '4GB', '128GB', 'Sedang', 'Ya'],
    ['Oppo', '4GB', '128GB', 'Sedang', 'Tidak'],
    ['iPhone', '4GB', '128GB', 'Sedang', 'Tidak'],
    ['Samsung', '8GB', '128GB', 'Mahal', 'Ya'],
    ['Xiaomi', '8GB', '128GB', 'Sedang', 'Ya'],
    ['Oppo', '8GB', '128GB', 'Mahal', 'Ya'],
    ['iPhone', '8GB', '128GB', 'Mahal', 'Ya']
]


# Data baru yang akan diprediksi.
test_data = [
    ['Samsung', '4GB', '64GB', 'Murah'],
    ['iPhone', '6GB', '128GB', 'Sedang'],
    ['Oppo', '6GB', '64GB', 'Murah'],
    ['Xiaomi', '4GB', '64GB', 'Murah'],
    ['iPhone', '4GB', '64GB', 'Murah']  # Kasus data tak terlihat
]

# Membuat daftar indeks atribut (0, 1, 2, 3) untuk proses training.
attributes = list(range(len(feature_names)))

# Membangun pohon keputusan dari dataset.
decision_tree = build_tree(dataset, attributes)

# Mencetak header untuk output pohon keputusan.
print("==== POHON KEPUTUSAN (Smartphone) ====\n")
# Mencetak struktur pohon keputusan yang sudah dibuat.
print_tree(decision_tree)

# Mencetak header untuk hasil prediksi.
print("\n==== HASIL PREDIKSI ====")
# Loop melalui setiap baris data tes.
for row in test_data:
    # Lakukan prediksi untuk baris tersebut.
    result = predict(decision_tree, row)
    # Cetak data input dan hasil prediksinya.
    print(f"Data: {row} -> Beli? {result}")
