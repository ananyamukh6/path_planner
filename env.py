import random, numpy as np, matplotlib.pyplot as plt, pdb, torch
from model import DQN, DQN2
import scipy.ndimage.filters as fi



def get_self_channel(w, h, loc):
    self_channel = 128*np.ones([w, h])
    rad = 5
    distdic = lambda a, b: {k:round(255 - (127)/(2.0*rad+1)) for k in range(2*rad+1)}[np.abs(i) + np.abs(j)]
    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            aa = loc[0] + i
            bb = loc[1] + j
            if aa >= h:
                aa = aa - h
            if aa < 0:
                aa = h + aa
            if bb >= w:
                bb = bb - w
            if bb< 0:
                bb = w + bb
            try:
                self_channel[aa, bb] = distdic(i,j)
            except:
                pdb.set_trace()
    return self_channel
                    
class Simulation(object):
    def __init__(self):
        self.grid = Grid()

class Agentlist(object):  #ensures all agents of the same type have the same model
    def __init__(self, agentlist, model):
        self.agentlist = agentlist
        self.model = model
    def __len__(self):
        return len(self.agentlist)
    def  __getitem__(self, idx):
        self.agentlist[idx]

class Agent(object):
    def __init__(self, alivecol, deadcol=None, loc=None):
        self.loc = loc
        self.alive = True
        self.alivecol = alivecol
        self.deadcol = alivecol if deadcol is None else deadcol
    def step(self, grid_state):
        assert False
    def reward(self):  #agent rewards itself, or grid does that?
        assert False
    def col(self):
        return self.alivecol if self.alive else self.deadcol


class ConstantSeeker(Agent):  #take fixed steps
    def step(self, grid_state):
        return np.array([1,1])
class RandomSeeker(Agent):  #take random steps
    def step(self, grid_state):
        return np.array([random.choice([1,0,-1]), random.choice([1,0,-1])])
class DedicatedSeeker(Agent):  #only move towards the nearest destination
    def step(self, grid_state):
        #pdb.set_trace()
        dest_locs = np.where(grid_state['gridmap']==1)
        diff = np.array(dest_locs) - np.array([[self.loc[0]], [self.loc[1]]])
        dsq = np.sum(diff*diff, 0)
        closest_dest = np.argmin(dsq)
        target_dest_w = dest_locs[0][closest_dest]; target_dest_h = dest_locs[1][closest_dest]
        stepw = 1 if target_dest_w > self.loc[0] else (-1 if target_dest_w < self.loc[0] else 0)
        steph = 1 if target_dest_h > self.loc[1] else (-1 if target_dest_h < self.loc[1] else 0)
        return np.array([stepw, steph])
class DQNSeeker(Agent):  #make sure to add a model to the class before usin it.
    def step(self, grid_state):
        w, h = grid_state['gridmap'].shape
        #agentmap = np.zeros([w,h])
        #for loc, agentclass in grid_state['agentinfo']:
        #    agentmap[loc[0], loc[1]] = agent_class_to_num[agentclass]
        #tmp = np.concatenate([grid_state['gridmap'][np.newaxis,...], agentmap[np.newaxis,...]], axis=0)  #2xwxh
        
        frame_history_len = 1
        #observations = np.zeros([1, w, h, frame_history_len*3])
        gridmap = np.copy(grid_state['gridmap'])
        gridmap[gridmap==0] = 128
        gridmap[gridmap==1] = 255
        gridmap[gridmap==1] = 0
        agentchannel = 128*np.ones(gridmap.shape)
        for kk in grid_state['agentinfo']:
            if kk[1] == Chaser:
                agentchannel[kk[0][0], kk[0][1]] = 0
                #TODO
                assert False, 'Do stuff present in else part (make agent a box instead of a point)'
            else:
                agentchannel[kk[0][0], kk[0][1]] = 255
                '''
                distdic = lambda a, b: {0:255, 1:220, 2:175}[np.abs(i) + np.abs(j)]
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        aa = kk[0][0] + i
                        bb = kk[0][1] + j
                        if aa >= h:
                            aa = aa - h
                        if aa < 0:
                            aa = h + aa
                        if bb >= w:
                            bb = bb - w
                        if bb< 0:
                            bb = w + bb
                        agentchannel[aa, bb] = distdic(i,j)
                '''

        #selfchannel = get_self_channel(w, h, self.loc)
        self_channel = 128*np.ones([w, h])
        self_channel[self.loc[0], self.loc[1]] = 255
        tt = np.stack([gridmap, agentchannel, self_channel])
        tt = tt/255.0
        tmp = torch.autograd.Variable(torch.FloatTensor(tt).unsqueeze(0))
        preds = self.model(tmp).cpu().data.numpy()[0]
        actn = np.argmax(preds)
        
        print actn, preds
        print 'XXXXXXXX'
        return action_to_delta[actn]

class Chaser(Agent):
    pass

agent_class_to_num = {ConstantSeeker:0, RandomSeeker:1, DedicatedSeeker:2, DQNSeeker:3, Chaser:4}

action_to_delta = {k:np.array([k//3, k%3])-1 for k in range(9)} #{0: array([-1, -1]), 1: array([-1,  0]), 2: array([-1,  1]), 3: array([ 0, -1]), 4: array([0, 0]), 5: array([0, 1]), 6: array([ 1, -1]), 7: array([1, 0]), 8: array([1, 1])}



class Grid(object):
    #cell type specs: 0: normal 1: destination, 2: pit
    def __init__(self, w, h, gridmap=None, grid_state_fn=None):
        self.w = w
        self.h = h
        self.gridmap = gridmap
        self.gridmap = self.init_grid() if gridmap is None else gridmap
        self.colmap = {0:(128,128,128), 1:(0,255,0), 2:(255,0,0)}
        self.agentlist = []
        self.shape = [h,w]
        self.n = 9
        self._grid_img = None
        self.num_steps = 0  #dont reset to 0 on env.reset()
        self.grid_state_fn = grid_state_fn
    def get_total_steps(self):
        return self.num_steps
    def reset(self):
        self.agentlist = []
        self._grid_img = None
        #self.num_steps = 0
        return self.get_obs_for_qlearning()
    def get_obs_for_qlearning(self):
        frame_history_len = 1
        observations = np.zeros([len(self.agentlist), self.w, self.h, frame_history_len*3])
        statedict = self.grid_state()
        gridmap = np.copy(statedict['gridmap'])
        gridmap[gridmap==0] = 128
        gridmap[gridmap==1] = 255
        gridmap[gridmap==1] = 0
        agentchannel = 128*np.ones(gridmap.shape)
        for kk in statedict['agentinfo']:
            if kk[1] == Chaser:
                agentchannel[kk[0][0], kk[0][1]] = 0
            else:
                agentchannel[kk[0][0], kk[0][1]] = 255
        for idx, kk in enumerate(statedict['agentinfo']):
            #self_channel = get_self_channel(self.w, self.h, kk[0])
            #self_channel = fi.gaussian_filter(self_channel, 3)
            self_channel = 128*np.ones([self.w, self.h])
            self_channel[kk[0][0], kk[0][1]] = 255
            observations[idx, :, :, 0] = gridmap
            observations[idx, :, :, 1] = agentchannel
            observations[idx, :, :, 2] = self_channel
        return observations
    def get_grid_img(self):
        if self._grid_img is None:
            disp_img = np.zeros([self.w, self.h, 3]).astype(np.uint8)
            for w in range(self.w):
                for h in range(self.h):
                    try:
                        disp_img[w,h,:] = self.colmap[self.gridmap[w,h]]
                    except:
                        pdb.set_trace()
            for a in self.agentlist:
                disp_img[a.loc[0],a.loc[1],:] = a.col()
            self._grid_img = disp_img
        assert self._grid_img is not None
        return self._grid_img
    def display(self):
        disp_img = self.get_grid_img()
        plt.imshow(disp_img); plt.show()
    def init_grid(self):
        num_destinations = random.choice([1,2,3,4,5])
        dest_dim = int(np.ceil(self.w/50.))
        gridmap = np.zeros([self.w,self.h])
        for k in range(num_destinations):
            tmpw = random.choice(range(0, self.w-dest_dim))
            tmph = random.choice(range(0, self.h-dest_dim))
            gridmap[tmpw:tmpw+dest_dim, tmph:tmph+dest_dim] = 1
        midw = int(np.round(self.w/2))
        midh = int(np.round(self.h/2))
        gridmap[midw-5:midw+5, midh-5:midh+5] = 2
        return gridmap
        #TODO: add random pits
        #Randomly populate the grid world with dangerous pits and destination cells
    def init_agents(self, agentlist): #register agents in the gridworld
        self.agentlist = agentlist
        self._grid_img = None
        for agent in self.agentlist:
            if agent.loc == None:
                while(True):
                    agent.loc = np.array([random.choice(range(self.w)), random.choice(range(self.h))])
                    if self.gridmap[agent.loc[0], agent.loc[1]] == 0:  #dont start agents at pits and destinations. start on a normal square only
                        break
    def get_agent_locs(self):
        for a in self.agentlist:
            print(a.loc)
    def clamp_loc(self):
        clamp_fn = lambda x, mn, mx : max(mn, min(x, mx))
        clamp_fn = lambda x, mn, mx : x%mx #toroidal world
        for a in self.agentlist:
            a.loc[0] = clamp_fn(a.loc[0], 0, self.w-1)
            a.loc[1] = clamp_fn(a.loc[1], 0, self.h-1)
    def grid_state(self):
        if self.grid_state_fn is None:
            return {'gridmap':self.gridmap, 'agentinfo':[(k.loc, k.__class__) for k in self.agentlist], 'grid_img':self.get_grid_img()}
        else:
            return self.grid_state_fn(self)
    def step(self, actionlist=None):
        #move agents
        if actionlist is not None:
            assert len(actionlist) == len(self.agentlist)
        else:
            grid_state = self.grid_state()
        for idx, agent in enumerate(self.agentlist):
            if agent.alive:
                if actionlist is None:
                    delta = agent.step(grid_state)
                else:
                    delta = action_to_delta[actionlist[idx]]
                agent.loc += delta


        self.clamp_loc()
        chasers_dict = {}
        seekers_dict = {}
        for idx, agent in enumerate(self.agentlist):
            if self.gridmap[agent.loc[0], agent.loc[1]] == 1:
                agent.alive = False  #agent fell into a pit
                if agent.__class__ == Chaser:
                    chasers_dict[idx] = {'pit':True, 'crash':False}
                else:
                    seekers_dict[idx] = {'pit':True, 'crash':False}
            if self.gridmap[agent.loc[0], agent.loc[1]] == 2:
                agent.alive = False #agent reached destination, so ending its lifecycle
                if agent.__class__ == Chaser:
                    #chasers should not be in destinations. its a pit for them
                    chasers_dict[idx] = {'pit':True, 'crash':False}
                else:
                    seekers_dict[idx] = {'pit':False, 'crash':False}

        #TODO: detect collision
        collisiondict = {}
        for idx, agent in enumerate(self.agentlist):
            if agent.alive:
                key = (agent.loc[0], agent.loc[1])
                collisiondict[key] = collisiondict.get(key, []) + [(idx, agent.__class__)]
        for k in collisiondict:
            if len(collisiondict[k]) > 1:
                chasers_in_same_cell_list = []
                seekers_in_same_cell_list = []
                for idx, cls in collisiondict[k]:
                    if self.gridmap[k[0], k[1]] == 0: #if you have reached destination cell, no fear of collisions, if in a pit, its penalized anyway
                        if cls is not Chaser:
                            seekers_in_same_cell_list += [idx]
                        else:
                            chasers_in_same_cell_list += [idx]
                #pdb.set_trace()            
                if (len(seekers_in_same_cell_list) > 1) or (len(seekers_in_same_cell_list) == 1 and len(chasers_in_same_cell_list) > 0): 
                    #more than 1 seeker, so a crash has occured or a chaser caught them     
                    for ppp in seekers_in_same_cell_list:
                        seekers_dict[ppp] = {'pit':False, 'crash':True}
                        self.agentlist[ppp].alive = False
                
                if len(chasers_in_same_cell_list) > 0:
                    if len(seekers_in_same_cell_list) > 0:
                        for ppp in chasers_in_same_cell_list:
                            self.agentlistp[ppp].alive = False  #The chasers caught some seekers, good for the chaser
                            chasers_dict[ppp] = {'pit':False, 'crash':False}
                    else:  #no seekers in this cell, so chasers crashed among themselves. bad
                        for ppp in chasers_in_same_cell_list:
                            self.agentlist[ppp].alive = False  #The chasers caught some seekers, good for the chaser
                            chasers_dict[ppp] = {'pit':False, 'crash':True}

                #for idx, cls in collisiondict[k]:
                #    if self.gridmap[k[0], k[1]] != 1:  #if you have reached destination cell, no fear of collisions
                        #print('collision')
                #        self.agentlist[idx].alive = False
                        
                        
                        
                        #if cls is not Chaser:
                        #    self.agentlist[idx].alive = False
                        #else:
                        #    chasers_in_same_cell_list += [idx]
                #if len(chasers_in_same_cell_list) > 1:
                #    #2 chasers collided. so they die
                #    for ii in chasers_in_same_cell_list:
                #        self.agentlist[ii].alive = False

        self._grid_img = None #invalidate current grid_img when a step has been taken (since it needs to be recalculated)

        #reward calculation
        reward = [0] * len(self.agentlist)
        for idx, ag in enumerate(self.agentlist):
            info = chasers_dict.get(idx, {'pit':False, 'crash':False}) if ag.__class__ == Chaser else seekers_dict.get(idx, {'pit':False, 'crash':False})
            status = ag.alive
            if not ag.alive:
                if not info['pit'] and not info['crash']:
                    reward[idx] = 1000  #success
                else:
                    reward[idx] = -1000 #death by crashing or pit
            else:
                reward[idx] = 0
            #TODO: a better reward calc is to reward based on dist from destination or penalize based on dist from other agents and pits

        self.num_steps += 1

        #pdb.set_trace()
        return [self.get_obs_for_qlearning(), reward, [not ag.alive for ag in self.agentlist]]





        #reward ideas:
        #collision avoidance: mean(dist to neighbours .. thresholded. above a certain threshold, it becomes constant)
        #each step taken: default reward of -1  (so that it tries to reduce the number of steps taken before reaching goal)
        #state is a 2channel thing: 1st channel is gridmap, second channel is agent locations on the map

        #using this loc the Grid updates the location.
        #In a non stochastic setting, the agents move to the cell they desire, in a stochastic setting, there is some noise added and agents may not land up exactly at the spot they wanted to move
        #reward agents, or kill them off
        #check if agents fell into pits, or collided, or reached destination



if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    if False:
        g = Grid(100,100)
        g.display()
        agentlist = [DedicatedSeeker((0,0,0), (255,255,255)) for k in range(50)]
        g.init_agents(agentlist)
        g.display()
        #pdb.set_trace()
        for k in range(100):
            g.step()
            g.display()
            #g.get_agent_locs()
            #print('xxxx')
        agl = Agentlist(agentlist, None)
        pdb.set_trace()
    else:
        gridmap = np.zeros([100,100])
        gridmap[40:60, 40:60] = 2
        gridmap[10:13,20:23] = 1
        gridmap[80:83,30:33] = 1
        gridmap[75:78,90:93] = 1
        g = Grid(100,100, gridmap)
        g.display()
        agentlist = [DQNSeeker((0,0,0), (255,255,255)) for k in range(50)]
        #model = DQN(3, 9).type(dtype)
        #model.load_state_dict(torch.load('/home/sayantan/Desktop/path_planner/path_planner/models/MyEnv1__5000_Mon_Dec_11_13:28:01_2017.model'))
        model = DQN2(2, 9).type(dtype)
        model.load_state_dict(torch.load('/home/sayantan/Desktop/path_planner/path_planner/models/MyEnv1__15000_Mon_Dec_11_14:34:00_2017.model'))
        for ag in agentlist:
            ag.model = model
        g.init_agents(agentlist)
        g.display()
        for k in range(100):
            g.step()
            g.display()
            #g.get_agent_locs()
            #print('xxxx')
        agl = Agentlist(agentlist, None)
        pdb.set_trace()