import torch
import numpy as np

class ODE_Task():
    def __init__(self, a):
        """
        :param a:
        :param z:
        """
        self.a = a

    def true_integral_val(self, total_samplesize=int(1e5)):

        z = np.random.normal(0, 1, total_samplesize)  # w_2
        a_rep = np.array(self.a.tolist() * total_samplesize)
        X = np.vstack((a_rep, z)).T  # x= (w_1, w_2); X: N by 2
        y = torch.tensor(self.ODE_Solver(X), dtype=torch.float)
        return y.mean().numpy()


    def ODE_Solver(self, X):
        '''
        this is the true_function
        :param X: \omega
        :return: fine approximation
        '''

        a = X[:, 0]
        z = X[:, 1]
        N = int(np.shape(a)[0])

        # set grid size
        n = 32

        # Step2: ODE solvers:
        Pf = np.zeros(N)  # fine approximation
        nf = n
        hf = 1 / nf
        l = 2

        if l > 1:
            # fine
            # A0f
            cf = np.repeat(1, nf)
            A0f = np.diag(-cf[1:] - cf[:(nf - 1)])
            grid = np.indices(((nf - 1), (nf - 1)))
            A0f[(grid[0] - grid[1] == 1) | (grid[0] - grid[1] == -1)] = 1
            A0f = hf ** (-2) * A0f

            # A1f
            cf = (np.array(range(1, (nf + 1))) - 0.5) * hf
            A1f = np.diag(-cf[1:] - cf[:(nf - 1)])
            grid = np.indices(((nf - 1), (nf - 1)))
            A1f[grid[0] - grid[1] == 1] = cf[1:nf - 1]
            A1f[grid[0] - grid[1] == -1] = cf[1:nf - 1]
            A1f = hf ** (-2) * A1f

            cf = np.repeat(1, nf - 1)

            for nl in range(0, N):
                U = a[nl]
                Z = z[nl]
                uf = np.linalg.inv(A0f + U * A1f) * (50 * Z ** 2 * cf)
                Pf[nl] = np.sum(hf * uf)

        return Pf





    def sample_data(self, size=1):
        """
        Sample data from this task.
        :returns:
            x: the feature vector of length size
            y: the target vector of length size
        """
        z = np.random.normal(0, 1, size)  # w_2
        a_rep = np.array(self.a.tolist() * size)
        W = np.vstack((a_rep, z)).T  # x= (w_1, w_2); X: N by 2

        y = self.ODE_Solver(W)

        X = torch.tensor(z, dtype=torch.float)
        X = X.squeeze().unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float)

        return X, y





class ODE_Task_Distribution():
    def __init__(self, a_min, a_max):
        # self.dim = dim
        self.a_min = a_min
        self.a_max = a_max

    def sample_task(self):
        a = np.random.uniform(self.a_min, self.a_max, 1)
        return ODE_Task(a)


