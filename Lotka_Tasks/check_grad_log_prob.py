import stan
import torch




# Step 1: Use Stan to compute grad_log_prob
mc = """
parameters {vector[2] x;}
model {x ~ multi_normal([0, 0], [[1, 0.8], [0.8, 1]]);}
"""
nchains = 4
nsams_perchain = 100
sm = stan.build(program_code=mc)
fit = sm.sample(num_chains=nchains, num_samples=nsams_perchain)

X = fit['x']
score_unconstrainedsamples = torch.zeros(nchains * nsams_perchain, 2)
for i in range(nchains * nsams_perchain):
    #  function 'unconstrain_pars' takes in constrained values and returns the corresponding unconstrained values.
    #         then the function 'grad_log_prob' will calculate the gradient of log posterior at the corresponing unconstrained values.
    temp = sm.grad_log_prob(sm.unconstrain_pars(dict(x=X[:, i].tolist())))
    score_unconstrainedsamples[i] = torch.tensor(temp)
score_unconstrainedsamples



# Step 1: Use closed form expression: grad_log_prob of multiGaussian
def grad_log_prob_multiGaussian(X, mu, Sigma_mat):
    n, d = X.size()[0], X.size()[1]
    out  = torch.zeros(n,d)
    for i in range(n):
        out[i,:] = torch.matmul(Sigma_mat.inverse(), (X[i,:] - mu).squeeze())
        # print(i,torch.matmul(Sigma_mat.inverse(), (X[i,:] - mu).squeeze()))
    out = -1.*out
    return out

mu = torch.zeros(2, dtype=torch.float64)
Sigma_mat = torch.tensor([[1, 0.8],[0.8, 1]], dtype=torch.float64)
Sigma_mat.size()
X  = torch.tensor(x, dtype=torch.float64)
X = X.t()
X.size()
X[:,0].squeeze().unsqueeze(1)
out = grad_log_prob_multiGaussian(X, mu, Sigma_mat)
out.size()

# Step 3: Check if both tensors are same
torch.allclose(out, score_unconstrainedsamples)
