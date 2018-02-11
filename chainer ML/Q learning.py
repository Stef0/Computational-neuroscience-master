import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import report, optimizers, Link, Chain
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TupleDataset
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions



class EvidenceEnv(object):
    """
    Very simple task which only requires evaluating present evidence and does not require evidence integration.
    The actor gets a reward when it correctly decides on the ground truth. Ground truth 0/1 determines probabilistically
    the number of 0s or 1s as observations
    """

    def __init__(self, n=1, p=0.8):
        """

        Args:
            n: number of inputs (pieces of evidence)
            p: probability of emitting the right sensation at the input
        """

        self.n_input = n
        self.p = p
        self.n_action = 2

        self._state = None

    def reset(self):
        """
        Resets state and generates new observations

        Returns:
            observation
        """

        # generate state
        self._state = np.random.choice(2)

        return self.observe()

    def step(self, action):
        """
        Executes action, updates state and returns an observation, reward, done (episodic tasks) and optional information

        :param action:
        :return: observation, reward, done, info
        """

        # return 1 for correct decision and -1 for incorrect decision
        reward = (2 * (action == self._state) - 1)

        # generate state
        self._state = np.random.choice(2)

        # we are always done after each decision
        done = True

        return self.observe(), reward, done, None

    def observe(self):
        """
        Helper function which generates an observation based on a state

        :return: observation
        """

        # generate associated observations
        P = [self.p, 1 - self.p] if self._state == 0 else [1 - self.p, self.p]

        return np.random.choice(2, self.n_input, True, P).astype('float32').reshape([1, self.n_input])[0]

    def render(self):
        """
        Takes care of rendering

        :return:
        """

        print self._state

    def close(self):
        """
        Closes the rendering

        :return:
        """
        pass

    def nout(self):
        """
        Closes the rendering

        :return:
        """
        reward0 = (2 * (0 == self._state) - 1)
        reward1 = (2 * (1 == self._state) - 1)

        
        return np.array([reward0, reward1])

    def asint(self,obs):
        """
        Represent input observations as an integer number
        :param obs:
        :return:
        """
        return int(sum(2**i*b for i, b in enumerate(obs)))

    def asbinary(self, i, b_len):
        """
        Represent integer as binary array
        :param i: integer
        :param b_len: length of binary array
        :return:
        """

        # get binary representation from integer
        _b = [int(x) for x in list('{0:0b}'.format(i))]
        _b = [0 for i in range(b_len - len(_b))] + _b

        return _b
    
    
    
class RandomAgent(object):
    def __init__(self, env):
        """
        Args:
        env: an environment
        """
        self.env = env
    def act(self, observation):
        """
        Act based on observation and train agent on cumulated reward (return)
        :param observation: new observation
        :param reward: reward gained from previous action; None indicates no reward because of initial state
        :return: action (Variable)
        """
        return np.random.choice(self.env.n_action)
    def train(self, a, old_obs, r, new_obs):
        """
        :param a: action
        :param old_obs: old observation
        :param r: reward
        :param new_obs: new observation
        :return:
        """
        pass    
    
    
class QAgent(object):
    def __init__(self, env):
        """
        Args:
        env: an environment
        """
        self.env = env
    def act(self, obs, Q):
        """
        Act based on observation and train agent on cumulated reward (return)
        :param observation: new observation
        :param reward: reward gained from previous action; None indicates no reward because of initial state
        :return: action (Variable)
        """
        if Q[self.env.asint(obs)][0] == Q[self.env.asint(obs)][1]:
            action = np.random.choice(2)
        elif  Q[self.env.asint(obs)][0] > Q[self.env.asint(obs)][1]:
            action = 0  
        else:
            action = 1
        return action
    def train(self, action, obs, reward, _obs, Q):
        
        Q[self.env.asint(obs), action] = Q[self.env.asint(obs), action] + 0.5 * ( reward + 0.5 * np.amax(Q[self.env.asint(_obs)]) - Q[self.env.asint(obs), action])
   
        
        
        """
        :param a: action
        :param old_obs: old observation
        :param r: reward
        :param new_obs: new observation
        :return:
        """
        pass  
      
#%%
   
# Number of iterations
n_iter = 1000
# environment specs
env = EvidenceEnv(n=2, p=0.95)
# define agent
agent = RandomAgent(env)
Qagent = QAgent(env)
# reset environment and agent
obs = env.reset()
reward = None
done = False
R = []
cum_rew=np.zeros(1000)
Q = np.zeros ([4,2])

'''random agen'''

for step in range(n_iter):
    env.render()
    action = agent.act(obs)
    _obs, reward, done, _ = env.step(action)
    # no training involved for random agent
    agent.train(action, obs, reward, _obs)
    
    obs = _obs
    R.append(reward)
    cum_rew [step:] += reward   # Cumulative reward
plt.plot(cum_rew)    





#%%
'''Qlearning'''


# Number of iterations
n_iter = 1000
# environment specs
env = EvidenceEnv(n=2, p=0.95)
# define agent
agent = RandomAgent(env)
Qagent = QAgent(env)
# reset environment and agent
obs = env.reset()
reward = None
done = False
R = []
cum_rew=np.zeros(1000)
Q = np.zeros ([4,2])


for step in range(n_iter):
    env.render()
    action = Qagent.act(obs, Q)
    _obs, reward, done, _ = env.step(action)
    
    Q[env.asint(obs), action] = Q[env.asint(obs), action] + 0.5 * ( reward + 0.5 * np.amax(Q[env.asint(_obs)]) - Q[env.asint(obs), action])
    
    obs = _obs
    R.append(reward)
    cum_rew [step:] += reward   # Cumulative reward
plt.plot(cum_rew)








'''
the Q matrix is 4*2 (4 possible states {1,0}* and 2 possible actions 1 or 0)

(these 4 possible states can be indexed from 0 to 3 with asint)

i fill it with zeros at the begginning
at each step I update it following:
    
select and carry out an action a (from act this will be 0 or 1)

observe reward r and new state s'  (reward and _obs)

Q[s,a] = Q[s,a] + α(r + γmaxa' Q[s',a'] - Q[s,a])


s = s'

THE AGENT is not acting randomly anymore:
it looks at the Q matrix, and then chooses the action that maximizes 
of course at the begginning the table is filled with zeors, so the agent, if the state has no prefferred action, 
chooses randomly

ALSO implement the Q learning in the train function and also update the agent class to be an agent that chooses
based on Q matrix  


'''



'''

MLP	 that	 takes	 observations	and	 learns	 to	 compute	 the	Q value	 for	all	 possible	actions.	
This	 is	done	by	backpropagation	on	the	loss	defined	by	the	sum	squared	difference	between	the	predicted	
and	desired	Q	values.	
For	simplicity	we	backpropagate	per	example.
NeuralQAgent	in	 chainer	which	 uses	an	MLP	 that	 takes	 observations	and	learns	 to	
compute	 the	Q value	 for	all	 possible	actions.	Use	 backpropagation	 to	 train	 your	 network.	 Plot	 the	
cumulative	rewards	for	this	agent.	Also	plot	the	values	for	Q(s,a)	before	and	after	learning. You	may	
use	the	function	EvidenceEnv.asbinary	(I,	b_len)	for	this.
'''
#%%

'''neuralagent'''

class neuralagent(Chain):
    def __init__(self):
        super(neuralagent, self).__init__()
        with self.init_scope():
            self.one = L.Linear(2, 5)  # the feed-forward output layer
            self.out = L.Linear(5,2)
            

    def __call__(self, obs):
        # Given the current word ID, predict the next word.
        hidden = self.one(obs)
        output = self.out(hidden)
        return output



class MyRegressor(chainer.Chain):
    def __init__(self, predictor):
        super(MyRegressor, self).__init__(predictor=predictor)

    def __call__(self, reward, Qpred, Qnew, action, gamma):
       
        
        # Outputs of the network is Qpred. For each of the 2 possible actions it has a value in the array Qpred.
        # Q-values can be any real values, which makes it a regression task, that can be optimized with a 
        # simple squared error loss. 
        
        loss = (reward + gamma*np.amax(Qnew._data) - Qpred[0][action])**2

        

        return loss


Neuralagent = neuralagent()
model = MyRegressor(Neuralagent)

optimizer = optimizers.SGD(lr = 0.1)
optimizer.setup(model)

# Number of iterations
n_iter = 1000
# environment specs
env = EvidenceEnv(n=2, p=0.95)
# define agent
agent = RandomAgent(env)
Qagent = QAgent(env)
# reset environment and agent
obs = env.reset()
reward = None
done = False
R = []
cum_rew=np.zeros(1000)
Q = np.zeros ([4,2])


for step in range(n_iter):
    env.render()
    
    obs = np.array(obs).reshape([1,2])
    
    if Neuralagent(obs)[0][0]._data > Neuralagent(obs)[0][1]._data :
    	action = 0
    else:
    	action = 1
    _obs, reward, done, _ = env.step(action)
       
    '''update of the Q : this is used in the loss calculation'''
    
    Qpred = Neuralagent(obs) #Qpredicted is used in the loss calculation
    _obs = np.array(_obs).reshape([1,2])  #needed for Qnew as it is the s'
    Qnew = Neuralagent(_obs)  #Qnew is used in the loss calculation it's the Q(s')
   
    
    model.cleargrads()
    loss = model(reward, Qpred, Qnew, action, 0.6)
    
    loss.backward()
    optimizer.update()
    
    obs = _obs
    R.append(reward)
    cum_rew [step:] += reward   # Cumulative reward
plt.plot(cum_rew)


    
Q[0, 0] = Neuralagent(np.array([0,0]).reshape([1,2]).astype('float32'))._data[0][0][0]
Q[0, 1] = Neuralagent(np.array([0,0]).reshape([1,2]).astype('float32'))._data[0][0][1]
Q[1, 0] = Neuralagent(np.array([1,0]).reshape([1,2]).astype('float32'))._data[0][0][0]
Q[1, 1] = Neuralagent(np.array([1,0]).reshape([1,2]).astype('float32'))._data[0][0][1]
Q[2, 0] = Neuralagent(np.array([0,1]).reshape([1,2]).astype('float32'))._data[0][0][0]
Q[2, 1] = Neuralagent(np.array([0,1]).reshape([1,2]).astype('float32'))._data[0][0][1]
Q[3, 0] = Neuralagent(np.array([1,1]).reshape([1,2]).astype('float32'))._data[0][0][0]
Q[3, 1] = Neuralagent(np.array([1,1]).reshape([1,2]).astype('float32'))._data[0][0][1]
    


'''

are both plot right (900-400 cumulative)?

WHY??
This tabular representation may be uncomputable for "real life" states:
If we wished to apply the Qlearning algorithm to learn from some images, for example, we would have to have a state
for each possible pixel color configuration. that would be an uncomputable number in terms of memory management.

EvidenceEnv.asbinary?

how to plot Qvalues?


'''