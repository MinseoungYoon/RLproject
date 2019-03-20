import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

## set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend() # aaa in [list] := list 안에 특정 원소가 존재하면 return True.
if is_ipython:
    from IPython import display

plt.ion()

## GPU
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


## Transition :: it maps (state,action) pairs to (next_state,reward) pairs
Transition = namedtuple('Transition', ('state' , 'action' , 'next_state' , 'reward'))

##
class ReplayMemory(object):

    def __init__(self , capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition"""
        if len(self.memory) < self.capacity :
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory , batch_size)

    def __len__(self):
        return len(self.memory)

##
class DQN(nn.Module):

    def __init__(self, h , w ):
        super(DQN , self).__init__()
        self.conv1 = nn.Conv2d(3 , 16 , kernel_size=5 , stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16 , 32 , kernel_size=5 , stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32 , 32 , kernel_size=5 , stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size , kernel_size=5 , stride=2):
            return (size - (kernel_size - 1) - 1)//stride + 1
        convW = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convH = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convW*convH*32

        self.head = nn.Linear(linear_input_size , 2)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0) , -1))


resize = T.compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor() ])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0) # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3 , but is somtimes larger
    # such as 800x1200x3(HxWxC). Transpose it into torch order (CxHxW).
    screen = env.render(mode='rgb_array').transpose((2,0,1))

    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height , screen_width = screen.shape

    screen = screen[: , int(screen_height * 0.4) : int(screen_height * 0.8)]

    view_width = int(screen_width * 0.6)

    cart_location = get_cart_location( screen_width )

    if cart_location < view_width // 2 :
        slice_range = slice(view_width)   # x = slice(start , end , step) => a[x] ::slice 된 내용 출력
    elif cart_location > ( screen_width - view_width//2 ):
        slice_range = slice(-view_width , None)
    else :
        slice_range = slice(cart_location - view_width //2 , cart_location + view_width //2)

    # Strip off the edges , so that we have a square image centered on a cart.
    screen = screen[ : , : , slice_range ]

    # Convert to float , rescale , convert to torch tensor
    # ( this doesn't require a copy )
    screen = np.ascontiguousarray( screen , dtype=np.float32 ) / 255
    screen = torch.from_numpy(screen)

    # Resize , and add a batch dimension (BxCxHxW)
    return resize(screen).unsqueeze(0).to(device) # torch. unsqueeze(0) :: add the new dimenstion at first dim


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy() , interpolation='none') # [np.array].squeeze(0) => 1-dimension인 0번째 차원을 삭제
plt.title('Example extracted screen')
plt.show()



##
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90.
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height , screen_width = init_screen.shape

policy_net = DQN(screen_height , screen_width).to(device)
target_net = DQN(screen_height , screen_width).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold :
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element wax found, so we pick action with the larget expected reward.
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor( [[random.randrange(2)]] , device=device , dtype=torch.long ) # random.randrange(2) :: 0 or 1


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor( episode_durations , dtype=torch.float )
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(duration_t) >= 100 :
        means = durations_t.unfold(0 , 100 , 1).mean(1).view(-1)   # torch.mean(1) :: 1 째 차원을 없애고 싶다 => (3,4).mean(1) => (3,)
        means = torch.cat((torch.zeros(99) , means))               # torch.unfold(dim , size , step) :: 특정 dim 을 따라size만큼씩 묶어나가는것.
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython :
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model() :
    ### TBC ###








