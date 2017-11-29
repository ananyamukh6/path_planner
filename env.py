import random, numpy as np, matplotlib.pyplot as plt, pdb

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
        agentmap = np.zeros([w,h])
        for loc, agentclass in grid_state['agentinfo']:
            agentmap[loc[0], loc[1]] = agent_class_to_num[agentclass]
        tmp = np.concatenate([grid_state['gridmap'][np.newaxis,...], agentmap[np.newaxis,...]], axis=0)  #2xwxh
        return model(tmp)  #todo, convert tmp to Variable and output of model to numpy

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
        self.shape = [h,w,3]
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
        return self.grid_state()
    def get_grid_img(self):
        if self._grid_img is None:
            disp_img = np.zeros([self.w, self.h, 3]).astype(np.uint8)
            for w in range(self.w):
                for h in range(self.h):
                    disp_img[w,h,:] = self.colmap[self.gridmap[w,h]]
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
        for agent in self.agentlist:
            if self.gridmap[agent.loc[0], agent.loc[1]] == 2:
                agent.alive = False  #agent fell into a pit

        #TODO: detect collision
        collisiondict = {}
        for idx, agent in enumerate(self.agentlist):
            if agent.alive:
                key = (agent.loc[0], agent.loc[1])
                collisiondict[key] = collisiondict.get(key, []) + [idx]
        for k in collisiondict:
            if len(collisiondict[k]) > 1:
                for idx in collisiondict[k]:
                    if self.gridmap[k[0], k[1]] != 1:  #if you have reached destination cell, no fear of collisions
                        print('collision')
                        self.agentlist[idx].alive = False
        self._grid_img = None #invalidate current grid_img when a step has been taken (since it needs to be recalculated)

        self.num_steps += 1





        #reward ideas:
        #collision avoidance: mean(dist to neighbours .. thresholded. above a certain threshold, it becomes constant)
        #each step taken: default reward of -1  (so that it tries to reduce the number of steps taken before reaching goal)
        #state is a 2channel thing: 1st channel is gridmap, second channel is agent locations on the map

        #using this loc the Grid updates the location.
        #In a non stochastic setting, the agents move to the cell they desire, in a stochastic setting, there is some noise added and agents may not land up exactly at the spot they wanted to move
        #reward agents, or kill them off
        #check if agents fell into pits, or collided, or reached destination



if __name__ == '__main__':
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