import scipy.io
from Sarcos.score_gamma import *

#  Load data
data = scipy.io.loadmat('Sarcos/sarcos.mat')


#  Get data and normalise data
X = torch.from_numpy(data['X'])
X = X.squeeze()
mu_X = X.mean(0).squeeze().unsqueeze(0)
std_X = X.std(0).squeeze().unsqueeze(0)
X.size()
mu_X.size()
std_X.size()
# Normalise X
X  = (X-mu_X)/std_X
X.size()



Xstar = torch.from_numpy(data['Xstar'])
Xstar = Xstar.squeeze()
Xstar.size()
# Normalise Xstar
Xstar  = (Xstar-mu_X)/std_X





y = torch.from_numpy(data['y']).squeeze()
mu_y = y.mean()
std_y = y.std()
# Normalise y
y = (y-mu_y)/std_y
y = y.unsqueeze(1)
print(y.size() , y.dtype)


ystar = torch.from_numpy(data['ystar']).squeeze()
# Normalise ystar
ystar = (ystar-mu_y)/std_y
ystar = ystar.unsqueeze(1)
print(ystar.size(), ystar.dtype)




# #  Save preprocessed data
torch.save(X, 'X.pt')
torch.save(y, 'y.pt')
torch.save(Xstar, 'Xstar.pt')
torch.save(ystar, 'ystar.pt')


