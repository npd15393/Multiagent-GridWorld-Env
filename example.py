from gw_multi_env import *

env=GridWorldMulti([5,6],2,[4,5,0,4])
agent1=Agent(0,env)
agent2=Agent(1,env)
currentState=env.reset()

while True:
	act1=agent1.randomAct()
	act2=agent2.randomAct()
	joint_act=act1+act2
	_,_,currentState,_,_= env.step(currentState, joint_act)