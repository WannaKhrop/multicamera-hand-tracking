import sys
import os

# Add 'src' directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src/utils'))
sys.path.append(src_path)
    

from utils import umeyama, linear_transfomation
import numpy as np

def main():

    # rotation + translation
    rot, trlt = np.random.normal(size=(3, 3)), np.random.normal(size=(3, 1))

    # make sure that it's a rotation matrix
    rot, _ = np.linalg.qr(np.random.normal(size=(3, 3)))

    # number of points
    N = 10

    # create points for check algorithms
    X_data = np.random.normal(size=(3, N))
    Y_data = np.dot(rot, X_data) + trlt

    # real transfomation matrix
    real_matrix = np.eye(4)
    real_matrix[:3] = np.hstack([rot, trlt])

    # call algorithms
    umeyama_result = umeyama(X_data.T, Y_data.T)
    mse_result = linear_transfomation(X_data, Y_data)

    # print all the results
    print('Umeyama:')
    print(umeyama_result)
    print()

    print('MSE:')
    print(mse_result)
    print()

    print('Real transformation:')
    print(real_matrix)
    print()

if __name__ == '__main__':
    main()