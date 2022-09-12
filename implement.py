from algorithms.XY_ADAPTIVE_ROBUST import XY_ADAPTIVE_ROBUST
from algorithms.XY_STATIC_ROBUST import XY_STATIC_ROBUST
import numpy as np


def sample_spherical(npoints, ndim=5):
    vec = np.random.rand(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def find_cloest_dist(X):
    min_dist_index = -1
    min_dis = 1e10

    for x1_idx, x1 in enumerate(X):
        for x2_idx, x2 in enumerate(X[x1_idx:]):
            if x1_idx == x2_idx: continue
            dis = np.linalg.norm(x1 - x2)
            if dis < min_dis:
                min_dis = dis
                min_dist_index = [x1_idx, x2_idx]
    return min_dist_index

def build_theta(X,alpha):
   min_dist_index = -1
   min_dis = 1e10

   for x1_idx, x1 in enumerate(X):
     for x2_idx, x2 in enumerate(X[x1_idx:,:]):
       if x1_idx == x2_idx: continue
       dis = np.linalg.norm(x1-x2)
       if dis < min_dis:
         min_dis = dis
         min_dist_index = [x1_idx, x2_idx]
   theta = X[min_dist_index[0]] + alpha * (X[min_dist_index[1]] - X[min_dist_index[0]])
   return theta
def build_Z(X,Y):
    """
    Y = [[Y(x1)],[Y(x2)]...]
    :return:
    """
    Z = []
    Z_index_list = []
    for x_index, x in enumerate(X):
        y_set = Y[x_index]
        for y_index, y in enumerate(y_set):
            Z.append(x - y)
            Z_index_list.append([x_index, y_index])
    Zhat = np.array(Z)
    return Zhat


def run_unit_sphere(num_arm):
    dim = 5
    alpha = 0.01
    res = []
    print("Number of arms: ", num_arm)

    # dim = 5
    # theta = sample_spherical(1,dim).reshape(dim, )
    # X = np.eye(dim)
    # tmp = np.zeros(dim)
    # tmp[0] = 1.0 + np.sin(0.01)
    # X = np.r_[X, np.expand_dims(tmp, 0)]
    X = sample_spherical(num_arm, dim).T
    min_ind = find_cloest_dist(X)
    theta = X[min_ind[0]] + alpha * (X[min_ind[1]] - X[min_ind[0]])

    Y = []
    num_y = 5
    for i in range(num_arm):
        Y_ = []
        for j in range(num_y):
            y_i = 0.01 * j * sample_spherical(1, dim).reshape(dim, )
            Y_.append(y_i)
        Y.append(Y_)
    # print(X)
    # print(np.array(Y))

    delta = 0.05
    alpha = 0.001
    xy = XY_ADAPTIVE_ROBUST(X, theta, alpha, delta, Y)
    # for i in range(10):
    #     xy.algorithm(i,True)

    res.append([xy.algorithm(dim, True)])
    return res

def run_unit_sphere1(num_arm, allocation):
    dim = 10
    res = []
    alpha = 0.01
    print("Number of arms: ", num_arm)

    # dim = 5
    # theta = sample_spherical(1).reshape(dim, )
    X = sample_spherical(num_arm,dim).T


    Y = []
    num_y = 5
    for i in range(num_arm):
        Y_ = []
        for j in range(num_y):
            y_i = 0.01 * j * sample_spherical(1,dim).reshape(dim, )
            Y_.append(y_i)
        Y.append(Y_)
    # print(X)
    # print(np.array(Y))
    Z = build_Z(X, Y)
    theta = build_theta(Z, alpha)

    delta = 0.05
    alpha = 0.0006
    xy = allocation(X, theta, delta, Y)
    # for i in range(10):
    #     xy.algorithm(i,True)

    res.append([xy.algorithm(dim)])
    return res


if __name__ == '__main__':
    res_adaptive = []
    for i in range(5, 35, 5):
        re = []
        for j in range(5):
            re.append(run_unit_sphere1(i, XY_STATIC_ROBUST))
        res_adaptive.append(re)
    # res_adaptive = []
    # for i in range(10, 60, 10):
    #     print(i)
    #     res_adaptive.append(run_unit_sphere(i))
