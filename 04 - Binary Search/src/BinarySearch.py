#Binary Search Algorithm
#Sintaks ini diperoleh melalui referensi Youtube
#Channel "Indonesia Belajar"

def binary_search(list, item):
    low = 0 # Nilai terendah
    high = len(list) - 1

    while low <= high:
        mid = (low + high) // 2 #Rumus nilai tengah
        guess = list[mid] #Mencari nilai tengah
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None

# Akhir Algoritma Binary Search