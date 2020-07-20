# using numpy to test if the pseudoinverse is correct

import numpy as np

A = np.array([[7, 2, 5, 9], [3, 4, 6, 23], [9, 8, 10, 14], [12, 6, 4, 3]])
A_plus = np.linalg.pinv(A)
print(A_plus)
