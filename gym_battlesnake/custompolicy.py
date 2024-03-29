import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.model import NNBase, Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)
        
class SnakePolicyBase(NNBase):
    ''' Neural Network Policy for our snake. This is the brain '''
    # hidden_size must equal the output size of the policy_head
    def __init__(self, num_inputs, recurrent=False, hidden_size=128): 
        super().__init__(recurrent, hidden_size, hidden_size)
        
        # We'll define a 3-stack CNN with leaky_relu activations and a batchnorm
        # here.
        self.base = nn.Sequential(
            nn.Conv2d(17, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(),
        )
        
        # Try yourself: Try different pooling methods
        # We add a pooling layer since it massively speeds up training
        # and reduces the number of parameters to learn.
        self.pooling = nn.AdaptiveMaxPool2d(2)
        
        # Try yourself: Change the number of features
        # 64 channels * 4x4 pooling outputs = 1024
        self.fc1 = nn.Linear(in_features=32*2*2, out_features=128)
        
        # Value head predicts how good the current board is
        self.value_head = nn.Linear(in_features=128, out_features=1)
        
        # Policy network gives action probabilities
        # The output of this is fed into a fully connected layer with 4 outputs
        # (1 for each possible direction)
        self.policy_head = nn.Linear(in_features=128, out_features=128)
        
        # Use kaiming initialization in our feature layers
        init_cnn(self)
        
    def forward(self, obs, rnn_hxs, masks):
        out = F.leaky_relu(self.base(obs))
        out = self.pooling(out).view(-1, 128)
        out = F.leaky_relu(self.fc1(out))
        
        value_out = self.value_head(out)
        policy_out = self.policy_head(out)
        
        return value_out, policy_out, rnn_hxs
    
class PredictionPolicy(Policy):
    """ Simple class that wraps the packaged policy with the prediction method needed by the gym """

    def predict(self, inputs, deterministic=False):
        # Since this called from our gym environment
        # (and passed as a numpy array), we need to convert it to a tensor
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        value, actor_features, rnn_hxs = self.base(inputs, None, None)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action, value
        
def create_policy(obs_space, act_space, base):
    """ Returns a wrapped policy for use in the gym """
    return PredictionPolicy(obs_space, act_space, base=base)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

