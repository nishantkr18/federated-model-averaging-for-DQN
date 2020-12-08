from dqn import *
from agent_compiler import *
from plotter import plot_graph

def test_agent(env, agent, runs=3):
    avg_score = []
    agent.is_test = True
    for i in range(runs):
        agent.is_test = True
        state = env.reset()
        done = False
        score = 0
        while not done:
            # env.render()
            action = agent.select_action(state)
            next_state, reward, done = agent.step(env, action)

            state = next_state
            score += reward
        avg_score.append(score)
    # print("Avg test score for {} runs: {}".format(
    #     runs, sum(avg_score)/len(avg_score)))
    return sum(avg_score)/len(avg_score)

def make_new_env(ENV_NAME, seed):
    env = gym.make(ENV_NAME)
    env.seed(seed)
    return env


if __name__ == "__main__":
    ENV_NAME = "CartPole-v0"
    torch.manual_seed(0)
    np.random.seed(0)

    # environment
    env = make_new_env(ENV_NAME, 0)
    test_env = make_new_env(ENV_NAME, 0)

    NO_OF_TRAINERS = 3
    FREQUENCY_OF_UPDATE = 30

    global_agent = DQNAgent(env)
    single_agent = DQNAgent(env, network=global_agent.dqn)

    agents = [DQNAgent(env, network=global_agent.dqn)
              for i in range(NO_OF_TRAINERS)]
    scores_single_agent = []
    scores_global_agent = []
    scores=[]
    steps = []
    for runs in range(1, 50000000):
        for (i, agent) in enumerate(agents):
            # training each agent serially (needs to be parallelized)
            agent.train(FREQUENCY_OF_UPDATE)
            # scores.append(test_agent(test_env, agent)) #Cartpole
            # scores.append(501+test_agent(test_env, agent)) #Acrobot
            # scores.append(201+test_agent(test_env, agent)) #Mountain Car
        single_agent.train(FREQUENCY_OF_UPDATE)

        # global_agent = combine_agents_reward_based(global_agent, agents, scores)
        global_agent = combine_agents(global_agent, agents)
        scores=[]
        agents = distribute_agents(global_agent, agents)


        if(runs%5==0):
            scores_global_agent.append(test_agent(test_env, global_agent))
            scores_single_agent.append(test_agent(test_env, single_agent))
            steps.append(single_agent.step_cnt)

            np.savetxt('arrays/scores_global_agent_'+ENV_NAME+'.csv', np.array(scores_global_agent))
            np.savetxt('arrays/scores_single_agent_'+ENV_NAME+'.csv', np.array(scores_single_agent))
            np.savetxt('arrays/steps_'+ENV_NAME+'.csv', np.array(steps))
            ###############PLOT##################
            plot_graph(scores_global_agent, scores_single_agent, steps, ENV_NAME, NO_OF_TRAINERS, ROLLING=20)


