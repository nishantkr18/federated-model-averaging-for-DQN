from dqn import *

def combine_agents(env, agents):
    main_agent = DQNAgent(env)
    
    for i in range(len(agents)):
        for main_param, agent_param in zip(main_agent.dqn.parameters(), agents[i].dqn.parameters()):
            if(i == 0):
                main_param.data.copy_(agent_param)
            else:
                main_param.data.copy_(main_param * (i/i+1) + agent_param * (1/i+1))

    return main_agent