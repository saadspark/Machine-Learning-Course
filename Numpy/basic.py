import numpy as np

my_list = [1, 2, 3, 4]

arr = np.array([[1, 2, 3, 4],[5,6,7,8]])
print(arr.ndim) #dimension of the array
print(arr.shape) #shape of the array
print(arr.size) #size of the array
print(arr.dtype) #data type of the array
print(arr.itemsize) #size of each element in the array
print(arr.nbytes) #total size of the array

print("----------------------Special arrays--------------------------------")

print(np.zeros((2,2)))   #accessing the first element of the array
print(np.ones((3,3)))   #accessing the first element of the array
print(np.empty((3,3)))   #accessing the first element of the array
print(np.arange(10))   #accessing the first element of the array
print(np.arange(1,10,2))   #accessing the first element of the array
print(np.arange(1,10,2))   #accessing the first element of the array

print("----------------------Array operations--------------------------------")
print(np.sum(arr))
print(np.mean(arr))
print(np.median(arr))
print(np.std(arr))
print(np.var(arr))
print(np.max(arr))

print("----------------------Indexing (2D arrays)--------------------------------")
mat = np.array([[1,2,3],
                [4,5,6],
                [7,8,9]])

print(mat[0,1])   # row 0, column 1 → 2
print(mat[2,2])   # row 2, column 2 → 9

print("----------------------Slicing (2D arrays)--------------------------------")
print(mat[0:2,1:3])   # row 0 to 2, column 1 to 3 → [[2 3] [5 6]]

test = np.array([5, 10, 15, 20, 25, 30])
print(test[1:4])   