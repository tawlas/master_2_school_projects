import argparse
# from script.interpolate import interp1d
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
value = parser.parse_args()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        """
        """
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 3)

    def forward(self, x):
        """
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Rossler_model:
    def __init__(self, delta_t):
        # super().__init__()
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = int(10000//self.delta_t)

        self.rosler_nn = Model().to(device)
        self.rosler_nn.load_state_dict(torch.load('model_trained.pth', map_location=device))
        self.initial_condition = np.array(value.init)

    def full_traj(self): 
        initial_condition=self.initial_condition
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary.
        y = np.zeros((self.nb_steps+1,3))
        w = initial_condition
        y[0] = w
        print("Predicting a trajectory from t=0 to t=10000 using a time step dt= :",self.delta_t,"starting at", initial_condition, "\n")
        for i in tqdm(range(1, self.nb_steps+1)):
            tens =  torch.Tensor(w).to(device)
            w = list(self.rosler_nn(tens).detach().cpu().clone().numpy())
            y[i]= w
        #if your delta_t is different to 1e-2 then interpolate y
        #in a discrete time array t_new = np.linspace(0,10000, 10000//1e-2)
        # y_new = interp1d(t_new, your_t, your_y)
        # I expect that y.shape = (1000000,)
        print("Shape of y", y.shape)
        return y

    def save_traj(self,y):
        #save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)
        outpath = 'traj.npy'
        np.save(outpath,y)
        print("Trajectory saved at:", outpath)
        
    
if __name__ == '__main__':

    ROSSLER = Rossler_model(delta_t=1e-2)
    print('--------Computing the trajectory----------')
    y = ROSSLER.full_traj()
    print('--------Saving the trajectory----------')
    ROSSLER.save_traj(y)
    # outpath = 'traj.npy'
    # y = np.load(outpath)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(y[:,0], y[:,1], y[:,2])
    plt.title('Predicted trajectory')
    plt.show()
    