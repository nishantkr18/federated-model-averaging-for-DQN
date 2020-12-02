from dqn import *

def combine_agents(main_agent, agents):
    for i in range(len(agents)):
        for main_param, agent_param in zip(main_agent.dqn.parameters(), agents[i].dqn.parameters()):
            if(i == 0):
                main_param.data.copy_(agent_param)
            else:
                main_param.data.copy_(main_param * (i/i+1) + agent_param * (1/i+1))
        for main_param, agent_param in zip(main_agent.dqn_target.parameters(), agents[i].dqn_target.parameters()):
            if(i == 0):
                main_param.data.copy_(agent_param)
            else:
                main_param.data.copy_(main_param * (i/i+1) + agent_param * (1/i+1))

    return main_agent

def distribute_agents(main_agent, agents):
    for i in range(len(agents)):
        for main_agent_param, agent_param in zip(main_agent.dqn_target.parameters(), agents[i].dqn_target.parameters()):
            agent_param.data.copy_(main_agent_param)
        for main_agent_param, agent_param in zip(main_agent.dqn.parameters(), agents[i].dqn.parameters()):
            agent_param.data.copy_(main_agent_param)
    return agents

def combine_agents_reward_based(main_agent, agents, scores):
    # import pdb
    # pdb.set_trace()
    total_reward=sum(scores)

    # parameters=main_agent.dqn.parameters()*0
    
    for i in range(len(agents)):
        for main_param, agent_param in zip(main_agent.dqn.parameters(), agents[i].dqn.parameters()):
            if i==0:
                main_param.data.copy_(agent_param*(scores[i]/total_reward))
            else:
                main_param.data.copy_(main_param+agent_param*(scores[i]/total_reward))
        for main_param, agent_param in zip(main_agent.dqn_target.parameters(), agents[i].dqn_target.parameters()):
            if i==0:
                main_param.data.copy_(agent_param*(scores[i]/total_reward))
            else:
                main_param.data.copy_(main_param+agent_param*(scores[i]/total_reward))

    return main_agent
