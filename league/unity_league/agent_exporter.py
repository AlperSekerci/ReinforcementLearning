import sys
if len(sys.argv) < 2:
    print("agent_name argument is missing")
    exit()

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

from envs.unity.my_unity.volleyball.volleyball import create_ppo_agent as create_agent
name = sys.argv[1]
agent = create_agent(train_on=False)
agent.restore(name)
agent.export(name)
