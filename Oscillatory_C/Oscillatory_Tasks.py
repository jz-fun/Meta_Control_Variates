
import torch
import numpy as np



class Oscillatory_Task():
    def __init__(self, a, u, xmin=0, xmax=1):
        """
        :param a:
        :param u:
        :param xmin: 0
        :param xmax: 1
        """
        self.a = a
        self.u = u
        self.xmin = xmin
        self.xmax = xmax

        self.dim = len(a)

    def true_integral_val(self):
        cst_deno = np.prod(self.a)
        if self.dim == 1:
            out = np.sin(2 * np.pi * self.u[0] + self.a[0]) / cst_deno - np.sin(2 * np.pi * self.u[0]) / cst_deno
            return out
        elif self.dim == 2:
            p1 = -1 * np.cos(2 * np.pi * self.u[0] + np.sum(self.a))/cst_deno
            p2 = -1 * np.cos(2 * np.pi * self.u[0] + self.a[1])/cst_deno + -1 * np.cos(2 * np.pi * self.u[0] + self.a[0])/cst_deno
            p3 = -1 * np.cos(2 * np.pi * self.u[0])/cst_deno
            out = p1-p2+p3
            return out
        else:
            X = np.random.uniform(self.xmin, self.xmax, (100000, self.dim))
            y = self.true_function(X)

            X = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            return y.mean().numpy()


    def true_function(self, X):
        n = X.shape[0]
        d = X.shape[1]
        out = np.zeros(n)
        for i in range(n):
            out[i] = np.cos(2 * np.pi * self.u[0] + self.a @ X[i])
        return np.reshape(out, (n, -1))


    def true_function_on_2axis(self, xaxis1s, xaxis2s):
        zs = np.cos(2 * np.pi * self.u[0] + self.a[0] * xaxis1s+  self.a[1] * xaxis2s)
        return zs




    def sample_data(self, size=1):
        """
        Sample data from this task.
        :returns:
            x: the feature vector of length size
            y: the target vector of length size
        """

        X = np.random.uniform(self.xmin, self.xmax, (size, self.dim))
        y = self.true_function(X)

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        return X, y





class Oscillatory_Task_Distribution():
    def __init__(self, dim, a_min, a_max, u_min, u_max, x_min=0, x_max=1):
        self.dim = dim
        self.a_min = a_min
        self.a_max = a_max
        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max

    def sample_task(self):
        a = np.random.uniform(self.a_min, self.a_max, self.dim)
        u = np.random.uniform(self.u_min, self.u_max, self.dim)
        return Oscillatory_Task(a, u, self.x_min, self.x_max)

