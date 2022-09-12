import numpy as np
import itertools
import logging
import time

# import warnings
# warnings.filterwarnings('error')

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


def randargmin(b, **kw):
    idxs = np.where(np.abs(b - b.min()) < 1e-20)[0]
    return np.random.choice(idxs)


def construct_v_l(v_l_1_set, v_l_2_set):
    v_l = []
    for i, d_i in enumerate(v_l_1_set):
        for j, d_j in enumerate(v_l_2_set):
            v_l.append(d_i - d_j)
    return v_l


def build_v_l(Y):
    v_l = []
    for arm_index, y_set in enumerate(Y):
        d_x = []
        for y_index, y in enumerate(y_set):
            d_y = []
            for y_index_prime, y_prime in enumerate(y_set):
                if y_index == y_index_prime: continue
                d_y.append(y - y_prime)
            d_x.append(d_y)
        v_l.append(d_x)

    return v_l


def find_dominate_y(arm_y, theta_est):
    y = np.array(arm_y)
    dominate_index = np.argmax(y @ theta_est)
    return dominate_index


def stopping_condition1(z1, v_l_1_list, z2, v_l_2_list, A_inv, theta_est, const, phase):
    l = z1 - z2
    d = l.shape[0] // 5
    v_l = construct_v_l(v_l_1_list, v_l_2_list)
    v_l_norm_max = []
    for i in v_l:
        res = i @ A_inv @ i
        v_l_norm_max.append(res)
    v_l_norm_max = max(v_l_norm_max)

    lower = (l.T @ A_inv @ l + v_l_norm_max) * const
    upper = l.T @ theta_est + min(np.array(v_l) @ theta_est)

    a = v_l_norm_max * const
    b = l.T @ theta_est
    c = min(np.array(v_l) @ theta_est)

    # if (l.T @ A_inv @ l + v_l_norm_max) * const >= l.T @ theta_est + np.exp(-phase*0.5/d) * min( np.array(v_l) @ theta_est):
    #     return True
    if (l.T @ A_inv @ l + v_l_norm_max) * const >= l.T @ theta_est + 1 / max(phase - 8, 1) * min(
            np.array(v_l) @ theta_est):
        return True
    else:
        return False


class XY_ORACLE_ROBUST(object):

    def __init__(self, X, theta_star, delta, Y, Z=None):

        self.X = X
        self.Y = Y
        if Z is None:
            self.Z = X
        else:
            self.Z = Z
        self.K = len(X)
        self.K_Z = len(self.Z)
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.build_Z()
        self.opt_arm = self.Zhat_index_list[np.argmax(self.Zhat @ theta_star)][0]
        self.opt_arm_z = np.argmax(self.Zhat @ theta_star)
        rewards = self.Zhat @ self.theta_star
        self.v_l = build_v_l(Y)
        self.gaps = np.max(rewards) - rewards
        self.gaps = np.delete(self.gaps, self.opt_arm_z, 0)
        self.delta = delta

    def algorithm(self, seed, binary=False):

        self.seed = seed
        np.random.seed(self.seed)

        self.arm_counts = np.zeros(self.K_z)
        self.N = 0
        self.build_L()
        self.A = np.zeros((self.d, self.d))
        self.b = np.zeros((self.d, 1))

        stop = False
        self.u = 1.35
        self.phase_index = 1

        design, rho = self.optimal_allocation()

        while True:

            self.delta_t = self.delta / (2 * self.phase_index ** 2 * self.K_Z)
            num_samples = int(np.ceil(self.u ** self.phase_index))
            # logging.info('num samples %s' % str(num_samples))

            allocation = np.random.choice(self.K_z, num_samples, p=design).tolist()
            allocation = np.array([allocation.count(i) for i in range(self.K_z)])
            # logging.info('allocation %s' % str(allocation))

            pulls = np.vstack([np.tile(self.Zhat[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])

            if not binary:
                rewards = pulls @ self.theta_star + np.random.randn(num_samples)
            else:
                rewards = np.random.binomial(1, pulls @ self.theta_star, (num_samples, 1))

            self.A += pulls.T @ pulls
            self.A_inv = np.linalg.pinv(self.A)
            self.b += (pulls.T @ rewards).reshape(self.d, 1)
            self.theta_hat = self.A_inv @ self.b

            best_idx = self.check_stop()

            if best_idx is None:
                pass
            else:
                stop = True

            self.phase_index += 1
            self.arm_counts += allocation
            self.N += num_samples

            # if self.N % 100000 == 0:
            # logging.info('\n\n')
            # logging.debug('arm counts %s' % str(self.arm_counts))
            # logging.info('total sample count %s' % str(self.N))
            # logging.info('\n\n')

            if stop:
                break

            self.phase_index += 1

        del self.b
        del self.A
        del self.A_inv
        # del self.Yhat
        self.success = (self.opt_arm == best_idx)
        print('Succeeded? %s' % str(self.success))
        print('Best arm:', best_idx)
        print('Sample complexity %s' % str(self.N))
        return self.arm_counts, self.N

    # def build_Y(self):
    #
    #     self.Yhat = self.Z[self.opt_arm, :] - self.Z
    #     self.Yhat = np.delete(self.Yhat, self.opt_arm, 0)
    #     self.Yhat = self.Yhat/self.gaps

    def build_Z(self):
        """
        Y = [[Y(x1)],[Y(x2)]...]
        :return:
        """
        Z = []
        Z_index_list = []
        for x_index, x in enumerate(self.X):
            y_set = self.Y[x_index]
            for y_index, y in enumerate(y_set):
                Z.append(x - y)
                Z_index_list.append([x_index, y_index])
        self.Zhat = np.array(Z)
        self.Zhat_index_list = Z_index_list
        self.K_z = len(self.Zhat)

    def build_L(self):

        self.Lhat = self.Zhat[self.opt_arm_z, :] - self.Zhat
        self.Lhat = np.delete(self.Zhat, self.opt_arm_z, 0)
        self.Lhat = self.Lhat / self.gaps.reshape(self.gaps.shape[0], 1)

    def optimal_allocation(self):

        design = np.ones(self.K_z)
        design /= design.sum()

        max_iter = 5000

        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.Zhat.T @ np.diag(design) @ self.Zhat)

            U, D, V = np.linalg.svd(A_inv)
            Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            newL = (self.Lhat @ Ainvhalf) ** 2
            rho = newL @ np.ones((newL.shape[1], 1))

            idx = np.argmax(rho)
            l = self.Lhat[idx, :, None]
            g = ((self.Zhat @ A_inv @ l) * (self.Zhat @ A_inv @ l)).flatten()
            g_idx = np.argmax(g)

            gamma = 2 / (count + 2)
            design_update = -gamma * design
            design_update[g_idx] += gamma

            relative = np.linalg.norm(design_update) / (np.linalg.norm(design))

            design += design_update

            if relative < 0.01:
                break

        idx_fix = np.where(design < 1e-5)[0]
        drop_total = design[idx_fix].sum()
        design[idx_fix] = 0
        design[np.argmax(design)] += drop_total

        return design, np.max(rho)

    def check_stop(self):

        stop = True

        arm = self.X[self.opt_arm, :, None]
        arm_y = self.Y[self.opt_arm]
        v_l_set_index = find_dominate_y(arm_y, self.theta_hat)
        v_l_set = self.v_l[self.opt_arm][v_l_set_index]
        z1 = arm - np.array(arm_y[v_l_set_index].reshape(self.d, 1))

        for arm_idx_prime in range(self.K):

            if self.opt_arm == arm_idx_prime:
                continue

            arm_prime = self.X[arm_idx_prime, :, None]
            arm_y_prime = self.Y[arm_idx_prime]
            v_l_set_index_prime = find_dominate_y(arm_y_prime, self.theta_hat)
            v_l_set_prime = self.v_l[arm_idx_prime][v_l_set_index_prime]
            z2 = arm_prime - np.array(arm_y_prime[v_l_set_index_prime].reshape(self.d, 1))

            const = np.sqrt(2 * np.log(1 / self.delta_t))

            if stopping_condition1(z1, v_l_set, z2, v_l_set_prime, self.A_inv, self.theta_hat, const, self.phase_index):
                stop = False
                break

        if stop:
            return self.opt_arm
        else:
            None
