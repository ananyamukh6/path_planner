import torch, pdb
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        tmp = 8
        self.fc1 = nn.Linear(in_features=tmp*tmp*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN2(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN2, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        tmp = 8
        self.fc1 = nn.Linear(in_features=tmp*tmp*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512+2, out_features=128)  #+2 for the locaion tuple
        self.fc3 = nn.Linear(in_features=128, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        inp = x[:,0:2,:,:]
        loc = x[:,2,:,:]
        loc_ = loc.cpu().data.numpy()
        aglocs = [np.argmax(loc_[k,...]) for k in range(loc_.shape[0])]
        aglocstup = [[int(loc/100), loc%100] for loc in aglocs]
        assert sum([loc_[i,k[0],k[1]] for i,k in enumerate(aglocstup)]) == loc_.shape[0]  #all should be '1'
        aglocstup = np.array(aglocstup)
        #aglocstup = torch.FloatTensor(aglocstup)
        aglocstup = Variable(torch.from_numpy(aglocstup)).type(dtype)
        #pdb.set_trace()
        x = self.relu(self.conv0(inp))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  #4096

        x = self.relu(self.fc1(x))
        conc = torch.cat([x,aglocstup], 1)
        x = self.relu(self.fc2(conc))
        x = self.fc3(x)
        return x

class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        tmp = 9
        self.fc1_adv = nn.Linear(in_features=tmp*tmp*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=tmp*tmp*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x






