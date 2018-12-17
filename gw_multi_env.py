import numpy as np
import matplotlib.pyplot as plt
import math
import  sys, time, random
import pandas as pd
import seaborn as sns
import pygame, sys, time, random
from pygame.locals import *
import itertools

###########################################################################

class Tile:
	# An object in this class represents a single Tile that
	# has an image

	# initialize the class attributes that are common to all
	# tiles.

	borderColor = pygame.Color('black')
	borderWidth = 4  # the pixel width of the tile border
	image1 = pygame.image.load('turtlebot.jpg')
	image2 = pygame.image.load('marvin.jpg')
	image3 = pygame.image.load('charger.jpg')
	size = (5,5)

	def __init__(self, x, y, obs, surface, tile_size = (100,100)):
		# Initialize a tile to contain an image
		# - x is the int x coord of the upper left corner
		# - y is the int y coord of the upper left corner
		# - image is the pygame.Surface to display as the
		# exposed image
		# - surface is the window's pygame.Surface object

		self.obs = obs
		self.origin = (x, y)
		self.tile_coord = (x//tile_size[0], y//tile_size[1])
		self.surface = surface
		self.tile_size = tile_size

	def grid2board(self,pos):
		board_coord=()
		for i in range(len(pos)//2):
			board_coord=board_coord+(pos[2*i], Tile.size[1]-1-pos[2*i+1])
		return board_coord

	def draw(self, pos, goal,idx=None):
		# Draw the tile.
		pos=self.grid2board(pos)
		goal=self.grid2board(goal)

		rectangle = pygame.Rect(self.origin, self.tile_size)
		if self.obs:
			pygame.draw.rect(self.surface, pygame.Color('black'), rectangle, 0)
		elif goal[0:2] == self.tile_coord:
			pygame.draw.rect(self.surface, pygame.Color('green'), rectangle, 0)
		elif goal[2:] == self.tile_coord:
			pygame.draw.rect(self.surface, pygame.Color('blue'), rectangle, 0)
		else:
			pygame.draw.rect(self.surface, pygame.Color('white'), rectangle, 0)
			if not idx==None:
				self.surface.blit(pygame.image.load(str(idx)+'.jpg'), self.origin)

		if goal[0:2] == self.tile_coord:
			self.surface.blit(Tile.image3, self.origin)

		if pos[0:2] == self.tile_coord:
			self.surface.blit(Tile.image1, self.origin)
		elif pos[2:] == self.tile_coord:
			self.surface.blit(Tile.image2, self.origin)

		pygame.draw.rect(self.surface, Tile.borderColor, rectangle, Tile.borderWidth)

#######################################################################################

class GridWorldMulti:
	def __init__(self,size=[8,8],n_agents=2,goal=(7,7,1,5))
		assert len(size)==2, "Grid has to be 2D"
		assert len(goal)==2*n_agents, "All agents have to have a goal"
		
		self.gridSize=size
		self.Pt=[[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
		self.Start=(0,0,4,0)
		self.n_agents=n_agents
		self.Goal=(7,7,1,5)
		self.obstacles=[(1,1),(2,2),(5,2),(6,7),(2,4),(3,5),(1,9)]
		self.gamma=0.9
		self.board = []
		self.actions=[(1,0),(-1,0),(0,1),(0,-1),(0,0)]
		self.action_probs={(1,0):[0.8,0,0.1,0.1,0.0],(-1,0):[0,0.8,0.1,0.1,0],(0,1):[0.1,0.1,0.8,0,0],(0,-1):[0.1,0.1,0,0.8,0],(0,0):[0,0,0,0,1]}
		self.build_states()
		self.build_joint_actions()
		self.build_rewards()

	# Creates state space
	def build_states(self):
		self.states=[]*self.n_agents

		for i in range(self.gridSize[0]):
			for j in range(self.gridSize[1]):
				for k in range(self.gridSize[0]):
					for l in range(self.gridSize[1]):
						s=(i,j,k,l)
						for na in range(self.n_agents):
							if self.checkState(s) and not self.checkGoal(s,na):
								self.states[na].append(s)


	# Creates  set of joint actions
	# Output: returns none. Creates List of joint action tuples 
	def build_joint_actions(self):
        joint_acts = list(itertools.permutations(self.actions * self.n_agents, self.n_agents))

        act = []
        for i in joint_acts:
            f = ()
            for j in i:
                f = f + j
            act.append(tuple(f))
        self.j_actions = list(set(act))

	def build_rewards(self):
		self.R={}
		for i in range(self.n_agents):
			for s in self.states[i]:
				for a in self.j_actions:
					n_s=tuple(np.array(s)+np.array(a))

					if self.checkBounds(n_s,i) and self.checkGoal(n_s,i):
						self.R[s,a,i]=1
					elif self.checkBounds(n_s,i) and self.checkCollision(n_s,i):
						self.R[s,a,i]=-1
					else:
						self.R[s,a,i]=-0.0

	# Checks for state
	def checkState(self,s):
		s=np.array(s)
		return (not self.checkCollision(s)) and (s[0]<self.gridSize[0] and s[1]<self.gridSize[1] and np.all(s>=0)) 

	# checks if any agent in state s has reached its goal
	def checkGoal(self,s,i=None):
		s=tuple(s)
		if i==None:
			return self.checkAllGoals(s)
		else:
			return (s[2*i:2*i+2] == self.Goal[2*i:2*i+2])

	# checks if all agents in state s reached their goal
	def checkAllGoals(self,s):
		s=tuple(s)
		flg=True
		for na in range(self.n_agents):
			flg=flg and (s[2*i:2*i+2] == self.Goal[2*na:2*na+2])
		return flg

	# checks if state s is wothin the defined gridworld
	def checkBounds(self,s,i=None):
		s=np.array(s)
		if i==None:
			flg=True
			for na in range(self.n_agents):
				 flg=flg and (s[2*n]<self.gridSize[2*n] and s[2*n+1]<self.gridSize[1] and np.all(s>=0)) 
			return flg
		else:
			n_ss=s[2*i:2*i+2]
			return (n_ss[0]<self.gridSize[0] and n_ss[1]<self.gridSize[1] and np.all(n_ss>=0)) 

	# checks if state s is is on obstacle or has mutually colliding agents
	def checkCollision(self,s,i=None):
		s=tuple(s)
		temp={}
		if i==None:
			flg=False
			for n in range(self.n_agents):
				flg=flg or (s[2*n:2*n+2] in self.obstacles) or (s[2*n:2*n+2] in temp)
				temp.append(s[2*n:2*n+2])
			return flg
		else:
			flg=False
			for n in range(self.n_agents):
				flg=flg or (s[2*n:2*n+2] in temp)
				temp.append(s[2*n:2*n+2])
			return flg or (s[2*i:2*i+2] in self.obstacles)


	# Get all possible next states for joint state s and joint action a
	# Output : List of neighbors, probabilities dictionary
	def get_nxt_states(self,s,a,i=None):
		n_s=list(s)
		if i==None:
			neighbor_gens=[]
			for n in self.n_agents:
				neighbor_gens=self.get_nxt_states(s,a[2*n:2*n+2],n)
			return combine_neighbors(neighbor_gens)
		else:
			n_s=s[2*i:2*i+2]
			neighbors={}
			for act in range(len(self.actions)):
				if self.action_probs[a][act]>0:
					n_s=tuple(np.array(s[2*i:2*i+2])+np.array(self.actions[act]))
					neighbors[tuple(n_s)]=self.action_probs[a][act]
		return neighbors

	# Combine neighbors of individual agents into joint state permutations
	# Input: [{agent1_neighbor_1:prob_1,..., agent1_neighbor_k:prob_k},{agent2_neighbor_1:prob_1,..., agent2_neighbor_k:prob_k},...]
	# Output: [joint_state_1,..., joint_st_k],[joint_prob_1,...joint_prob_k]
	def combine_neighbors(self,dict_lst):
		# l1=list(dict1.keys())
		# l2=list(dict2.keys())
		lst=[]
		for n in range(self.n_agents):
			lst.append(dict_lst[n].keys())

		combos=tuple(list(lst[0]))

		for n in range(1:self.n_agents):
			comb=[zip(x,lst[n]) for x in itertools.permutations(combos,len(lst[n]))]

			chain = itertools.chain(*comb)
			combos = list(chain)
			for st in range(len(combos)):
				combos[st]=combos[st][0]+combos[st][1]
			combos=list(set(combos))
		
		probs={}
		for st in range(len(combos)):
		  probs[combos[st]]=1
		  for n in range(self.n_agents):
		    probs[combos[st]]*=dict_lst[n][combos[st][2*n:2*n+2]]

		return combos,probs

	# Move from state s with action a following stochastic dynamics
	# Collisions and going outside boundary keep agent in same state
	# Output: Next state
	def stochastic_trans(self,s,a,i=None):
		n_s=list(s)
		if i==None:
			ns=[]
			for n in range(self.n_agents):
				n_s=ns+stochastic_trans(self,s,a[2*n:2*n+2],n)[2*n:2*n+2]
		else:
			index = np.random.choice(range(len(self.actions)), p=self.action_probs[a])
			n_s[2*i:2*i+2]=tuple(np.array(s[2*i:2*i+2])+np.array(self.actions[index]))

		if self.checkBounds(n_s) and not self.checkCollision(n_s,i):	# checkAction(s,a,i) and not checkCollision(n_s,i):
			return tuple(n_s)
		else: return tuple(s)

	# Update pygame state 
	def renderEnv(self,s,a=None):

		for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					sys.exit()

		self.draw(s,a)


		# Refresh the display
		pygame.display.update()

		# Set the frame speed by pausing between frames
		time.sleep(pauseTime)


	# Draw pygame elements
	def draw(self,s,agent=None):		
		# Draw the tiles.
		# self is the Grid_World game
		pos=s
		goal = self.Goal
		self.surface.fill(pygame.Color('black'))
		idx=None
		for y in range(self.gridSize[1]):
			for x in range(self.gridSize[0]):
				row=self.board[y]
				row[x].draw(pos, goal, idx)

	# Execute joint action in state s
	# Output: [previous state, joint_action, next state, reward, is task complete]
	def step(self,s,a):
		j_act=()
		R=[]
		for n in range(self.n_agents):
			if self.checkGoal(s,n):
				j_act=j_act+(0,0)
				R.append(0)
			else:
				j_act=j_act+a[2*n:2*n+2]
				R.append(self.R[n][s,a])

		print('act '+str(act)+' st '+str(s))
		prev_st=s

		self.renderEnv(s)
		s=self.stochastic_trans(s,act)

		isDone=self.checkAllGoals(s)

		return [prev_st,act,s,R,isDone]

	def reset(self):
		return self.Start

	# Object mgmt for pygame sim
	def createTiles(self):
		# Create the Tiles
		# - self is the Grid_World game

		self.board = []
		for yIndex in range(self.gridSize[1]):
			row = []
			for xIndex in range(self.gridSize[0]):
				imageIndex = yIndex * self.gridSize[1] + xIndex
				y = (self.gridSize[1] -1- yIndex) * 100
				x = xIndex * 100
				if (xIndex,yIndex) in self.obstacles:
					wall = True
				else:
					wall = False
				tile = Tile(x, y, wall, self.surface)
				row.append(tile)
			self.board.append(row)

	# Plots value function of agent idx given other agents in state opp_s 
	def plot_heatmap(self,idx,opp_s=(0,0)):
		"""
		This function will plot the heatmap of the value of each state
		:param agent: The learning rate
		:return: NULL
		"""
		uniform_data = np.zeros([self.gridSize[0], self.gridSize[1]])
		for s in self.states[idx]:
			if s[0:2*(idx)]==opp_s[0:2*(idx)] and s[2*(idx)+2:]==opp_s[2*(idx)+2:]:
				uniform_data[s[2*idx+1]][s[2*idx]] = 1+V[s,idx]  # change the y to x, and x to y


		values = uniform_data
		values = np.flip(values, axis=0)
		index = np.arange(0, gridSize, 1)  # y axis
		index = reversed(index)  # the y coordinate index
		columns = np.arange(0, gridSize, 1)

		df = pd.DataFrame(values, index=index, columns=columns)

		ax = sns.heatmap(df, linewidths=.5)
		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_title('The Value function for agent '+str(idx)+' when agent '+str(1-idx)+' at '+str(opp_s))

		plt.show()


# Example agent class
class Agent:
	def __init__(self, idx):
		self.idx=idx
		self.Q={}
		self.env=GridWorldMulti([5,6],3,[4,5,])

	# Weighted Policy Update for agent i  
	def build_policy(self,i):
		maxchg=0
		for s in states[i]:
			# prev=pi[s,i]
			# pi[s,i]=0.3*pi[s,i]+0.7*getPolicy(s,i)
			# maxchg=max(maxchg,np.linalg.norm(pi[s,i]-prev))
		return maxchg

	# Make softmax policy for agent i in state s
	def getPolicy(self, s, det=None):
		"""
		This function will return an action given a stationary policy given the current state.
		:param state: The current state
		:param policy: The current policy
		:return: The action
		"""
		if det==None: det=False
		policy=[]

		for a in range(len(actions)): 
			targUtil=self.Q[s,self.idx][a]
			policy.append(math.exp(targUtil))

		policy=[p/sum(policy) for p in policy]
		return np.array(policy)

	# Collect data train
	def train(self):
		data=[]
		current_state=env.reset()
		# Collect data
		
		isDone=False
		while not isDone:
			act=env.j_actions[int(random.random()*len(env.j_actions))]
			data.append(env.step(current_state,act))

		# Process data 
		# Learn




