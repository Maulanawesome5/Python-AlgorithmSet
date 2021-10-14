#Memakai algoritma binary search

import BinarySearch

# 1. Mencari jumlah angka yang sama dalam list
my_list = [1, 3, 5, 7, 9]
print(
"""
Jumlah angka yang ada didalam list
""", BinarySearch.binary_search(my_list, 3)
) #Import fungsi


# 2. Mencari element/angka yang tidak ada dalam list
print(
"""
Apakah angka -1 ada didalam list ?
""", BinarySearch.binary_search(my_list, -1)
)
