import numpy as np
from forban import *
from forban.bandits import NormalBandit, Normal
from forban.sequentialg import SequentiAlg
from forban.utils import *
import matplotlib.pyplot as plt


# Create distributions
mean_1 = 0.
mean_2 = 1.
normal_1 = Normal(mean_1)
normal_2 = Normal(mean_2)
sample_1 = normal_1.sample()
sample_2 = normal_2.sample()
means = [mean_1, mean_2]
print(f"A sample from a normal distribution with mean {mean_1} and standard deviation 1 is {sample_1:.3f}")
print(f"A sample from a normal distribution with mean {mean_2} and standard deviation 1 is {sample_2:.3f}")

# Create bandit problem
bandit_instance = NormalBandit([0.1, 0.2, 0.32, 0.24, 0.22])
print(f"A bandit instance is {bandit_instance}")
plot_bandit(bandit_instance)


# Sequential Algorithms #

# Constantly play the same arm
class Constant(SequentiAlg):
    def __init__(self, bandit, name="Constant", params={'init': 1, 'choice': 0 }):
        SequentiAlg.__init__(self, bandit, name=name, params=params)
        self.choice = params['choice']
        self.name = f"{self.name} (arm {self.choice})"
        assert self.choice < bandit.nbr_arms, f"'choice' ({self.choice}) should be one of\
the arms indices (<{bandit.nbr_arms})"
    
    def compute_indices(self):
        self.indices[self.choice] = 0
        
# Non-optimized Explore Then Commit strategy        
class Etc(SequentiAlg):
    def __init__(self, bandit, name="ETC", params={'init': 0, 'exploration': 200 }):
        SequentiAlg.__init__(self, bandit, name=name, params=params)
        self.exploration = params['exploration']
        
    def compute_indices(self):
        if self.time <= self.exploration:
            # probably faster computation is possible
            self.indices = np.random.rand(self.bandit.nbr_arms)
        else:
            self.indices = self.means
            
    def choose_an_arm(self):
        return randamax(self.indices)
    
    
    
# Experiment
# Set of means
means = [0.2, 0.5, 0., 0.05, 0.3, 0.4]
# Create a bandit problem
bandit = NormalBandit(means)
# Create sequential algorithms
seqalg1 = Constant(bandit) # Constantly play the same arm
seqalg2 = SequentiAlg(bandit, name="Random") # Sequentially choose an arm using a uniform distribution on the set of arms
seqalg3 = Constant(bandit, params = {'init': 1, 'choice': 3}) # Constantly play the same arm
seqalg4 = Etc(bandit) # Explore Then Commit strategy

print(seqalg4)

test_experiment = Experiment([seqalg1, seqalg2, seqalg3, seqalg4], bandit,
                             statistics={'mean':True, 'std':True, 'quantile':True, 'pulls':False},
                             complexity=False)

test_experiment.run(50, 550)
test_experiment.plot()

print(seqalg4)




# PART 2 #
class IMED(SequentiAlg):
    def __init__(self, bandit, name="IMED", params={'init': -np.inf, 'kl':klGaussian}):
        SequentiAlg.__init__(self, bandit, name=name, params=params)
        self.kl = params['kl']
    
    def compute_indices(self):
        max_mean = np.max(self.means)
        if self.all_selected:
            self.indices = self.nbr_pulls*self.kl(self.means, max_mean) + np.log(self.nbr_pulls)
        else:
            for arm in np.where(self.nbr_pulls != 0)[0]:
                self.indices[arm] = self.nbr_pulls[arm]*self.kl(self.means[arm], max_mean) \
                + np.log(self.nbr_pulls[arm])
                
                
                
horizon = 700
agent = IMED(bandit)
experiment_imed = Experiment([agent], bandit,
                             statistics={'mean':True, 'std':False, 'quantile':False, 'pulls':False},
                             complexity=False)

experiment_imed.run(1, horizon)
experiment_imed.plot()
# Visualize results (on one run) 

# Histogram of the number of arms selections
plt.figure(figsize=(12,5))
plt.xlabel("Arms", fontsize=14)
plt.ylabel("Number of arms selections", fontsize=14)
plt.bar(range(bandit.nbr_arms), agent.nbr_pulls, width=0.4, tick_label=[str(i) for i in range(bandit.nbr_arms)])
plt.title("Number of selections of each arm", fontsize=14)
plt.show()


nbr_exp = 50
horizon = 500
etc = Etc(bandit) # Explore Then Commit strategy
imed = IMED(bandit) # IMED strategy
experiment = Experiment([etc, imed], bandit,
                        statistics={'mean':False, 'std':False, 'quantile':True, 'pulls':False},
                        complexity=True)
experiment.run(nbr_exp, horizon)
experiment.plot()