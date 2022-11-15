# Python Program illustrating
# numpy.sum() method
import numpy as np
      
# 1D array
arr = [20, 2, .2, 10, 4] 
  
print("\nSum of arr : ", np.sum(arr))
  
print("Sum of arr(uint8) : ", np.sum(arr, dtype = np.uint8))
print("Sum of arr(float32) : ", np.sum(arr, dtype = np.float32))
  
print ("\nIs np.sum(arr).dtype == np.uint : ",
       np.sum(arr).dtype == np.uint)
 
print ("Is np.sum(arr).dtype == np.float : ",
       np.sum(arr).dtype == np.float)


print("Sum of arr(float32) : ", np.sum(arr, dtype = np.float32))