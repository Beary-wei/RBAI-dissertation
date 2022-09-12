import numpy as np
import itertools
import logging

DIM = 5

def randargmin(b, **kw):
    idxs = np.where(np.abs(b - b.min()) < 1e-20)[0]
    return np.random.choice(idxs)

def upper_bound(z, theta_est, A_inv, const):
    return z.T @ theta_est + const * z.T @ A_inv @ z

def lower_bound(z, theta_est, A_inv, const):
    return z.T @ theta_est - const * z.T @ A_inv @ z
def stopping_condition(x1,y1_set,x2,y2_set,theta_est,A_inv,const):
    """

    :param x1: x_star
    :return:
    """
    z1_set = [upper_bound(x1 - y1.reshape(y1.shape[0], 1), theta_est, A_inv, const) for y1 in y1_set]
    z2_set = [lower_bound(x2 - y2.reshape(y2.shape[0], 1), theta_est, A_inv, const) for y2 in y2_set]

    if min(z1_set) <= min(z2_set):
        return True
    else:
        return False

def find_dominate_y(arm_y, theta_est):
    y = np.array(arm_y)
    dominate_index = np.argmax(y@theta_est)
    return dominate_index




class XY_ADAPTIVE_ROBUST(object):
    def __init__(self, X, theta_star, alpha, delta, Y):
        self.X = X
        self.Y = Y
        self.K = len(X)
        self.d = X.shape[1]
        self.theta_star = theta_star
        # self.opt_arm = np.argmax(X @ theta_star)
        # self.opt_arm = 0
        self.alpha = alpha
        self.delta = delta

    def algorithm(self, seed, verbose=False):

        np.random.seed(seed)

        self.active_arms = list(range(self.K))
        self.arm_counts = np.zeros(self.K)
        self.build_Z()
        self.Z_space = int(self.K_z)
        self.opt_arm = self.Zhat_index[np.argmax(self.Zhat @ self.theta_star)][0]

        n_past = self.d * (self.d + 1) + 1
        rho = 1
        rho_past = 1
        self.N = 0
        self.phase_index = 1
        self.build_L()
        while len(self.active_arms) > 1:

            n = self.d
            self.N += self.d
            self.A = np.eye(self.d)
            self.A_inv = np.linalg.inv(self.A)
            self.b = np.zeros((self.d, 1))

            while rho / n >= self.alpha * rho_past / n_past:
                arm_idx, rho = self.select_greedy_arm()
                arm = self.Zhat[arm_idx, :, None]

                r = self.pull(arm)
                self.b += arm * r
                self.A += arm @ arm.T
                self.A_inv -= (self.A_inv @ arm @ arm.T @ self.A_inv) / (1 + arm.T @ self.A_inv @ arm)

                n += 1
                self.arm_counts[self.Zhat_index[arm_idx][0]] += 1
                # TODO: add adversary count
                self.N += 1

            n_past = n
            rho_past = rho

            self.theta_hat = self.A_inv @ self.b
            self.drop_arms()
            self.build_Z()
            self.build_L()
            self.phase_index += 1

            # logging.info('\n\n')
            # logging.info('finished phase %s' % str(self.phase_index - 1))
            # logging.info('arm counts %s' % str(self.arm_counts))
            # logging.info('round sample count %s' % str(n))
            # logging.info('total sample count %s' % str(self.N))
            # logging.info('active arms %s' % str(self.active_arms))
            # logging.info('rho %s' % str(rho))
            # logging.info('\n\n')
            if verbose:
                print('\n\n')
                print('finished phase %s' % str(self.phase_index - 1))
                print('arm counts %s' % str(self.arm_counts))
                print('round sample count %s' % str(n))
                print('total sample count %s' % str(self.N))
                print('active arms %s' % str(self.active_arms))
                print('rho %s' % str(rho))
                print('\n\n')

        del self.b
        del self.A
        del self.A_inv
        self.success = (self.opt_arm in self.active_arms)
        # logging.critical('Succeeded? %s' % str(self.success))
        # logging.critical('Sample complexity %s' % str(self.N))
        print('Succeeded? %s' % str(self.success))
        print('Sample complexity %s' % str(self.N))
        # print("Active arm" % str(self.active_arms))
        if not self.success:
            print("Active arm:", self.active_arms)


    def build_Z(self):
        """
        Y = [[Y(x1)],[Y(x2)]...]
        :return:
        """
        Z = []
        Z_index = []
        for x_index, x in enumerate(self.X[self.active_arms]):
            y_set = self.Y[x_index]
            for y_index, y in enumerate(y_set):
                Z.append(x - y)
                Z_index.append([x_index, y_index])
        self.Zhat = np.array(Z)
        self.Zhat_index = Z_index
        self.K_z = len(self.Zhat)

    def build_L(self):

        L = []
        for z, z_prime in itertools.permutations(self.Zhat, 2):
            L.append(z - z_prime)


        self.Lhat = np.array(L)

        if self.N > 0:
            self.Lhat = np.append(self.Lhat, self.Zhat, axis=0)

    def select_greedy_arm(self):

        score = np.diag(self.Lhat @ self.A_inv @ self.Lhat.T).reshape(1, -1) \
                - ((self.Zhat @ self.A_inv @ self.Lhat.T) ** 2 / (
                1 + np.diag(self.Zhat @ self.A_inv @ self.Zhat.T).reshape(-1, 1)))

        arm_idx = randargmin(score[np.arange(self.K_z), np.argmax(score, axis=1)])

        rho = np.max(score[arm_idx, :])

        return arm_idx, rho

    def pull(self, arm):

        r = arm.T @ self.theta_star + np.random.randn()

        return r

    def drop_arms(self):

        active_arms = self.active_arms.copy()

        for arm_idx in active_arms:

            arm = self.X[arm_idx, :, None]
            arm_y = self.Y[arm_idx]

            for arm_idx_prime in active_arms:

                if arm_idx == arm_idx_prime:
                    continue

                arm_prime = self.X[arm_idx_prime, :, None]
                arm_prime_y = self.Y[arm_idx_prime]
                # const = np.sqrt(2 * np.log(2 * np.pi ** 2 * self.K_z ** 2 * self.phase_index ** 2 / (
                #         6 * self.delta)))
                const = np.sqrt(2 * np.log(2 * self.N ** 2 * self.K_z  / (
                        np.pi ** 2 * self.delta)))

                if stopping_condition(arm,arm_y,arm_prime,arm_prime_y, self.theta_hat,self.A_inv,const):
                    self.active_arms.remove(arm_idx)
                    break
                # if np.sqrt(2 * y.T @ self.A_inv @ y * np.log(2 * np.pi ** 2 * self.K ** 2 * self.phase_index ** 2 / (
                #         6 * self.delta))) <= y.T @ self.theta_hat:
                #     self.active_arms.remove(arm_idx)
                #     break


if __name__ == '__main__':
    def run(dim):
        print('Dim: ', dim)

        # dim = 5
        theta = np.zeros(dim)
        theta[0] = 1.0
        X = np.eye(dim)
        tmp = np.zeros(dim)
        tmp[0] = 1.0 + np.sin(0.01)
        X = np.r_[X, np.expand_dims(tmp, 0)]

        Y = []
        num_y = 5
        for i in range(dim + 1):
            Y_ = []
            for j in range(num_y):
                y_i = np.zeros(dim)
                if i == dim:
                    y_i[0] = 0.01 * (j + 1) + 2 * np.sin(0.01)
                else:
                    y_i[i] = 0.01 * (j + 1)
                Y_.append(y_i)
            Y.append(Y_)
        # print(X)
        # print(np.array(Y))

        delta = 0.05
        alpha = 0.1
        xy = XY_ADAPTIVE_ROBUST(X, theta,alpha, delta, Y)
        # for i in range(10):
        #     xy.algorithm(i,True)
        xy.algorithm(dim,True)


    for i in range(2, 6):
        run(i * 5)