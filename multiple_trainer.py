from dqn import *
from agent_compiler import *


def test_agent(env, agent, runs=5):
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
    print("Avg test score for {} runs: {}".format(
        runs, sum(avg_score)/len(avg_score)))
    return sum(avg_score)/len(avg_score)


if __name__ == "__main__":
    ENV_NAME = "LunarLander-v2"
    # environment
    envs = []
    main_env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    NO_OF_TRAINERS = 3
    for i in range(NO_OF_TRAINERS):
        envs.append(gym.make(ENV_NAME))

    main_agent = DQNAgent(main_env)

    agents = [DQNAgent(envs[i], network=main_agent.dqn)
              for i in range(NO_OF_TRAINERS)]
    scores_agent = []
    scores_main_agent = []
    for runs in range(1, 5000):
        for (i, agent) in enumerate(agents):
            # training each agent serially (needs to be parallelized)
            agent.train(20)
            # scores.append(test_agent(test_env, agent))
            if(i == 0):
                scores_agent.append(test_agent(test_env, agent))

        # main_agent = combine_agents_reward_based(main_agent, agents, scores)
        main_agent = combine_agents(main_agent, agents)
        print('Testing main_agent')

        scores_main_agent.append(test_agent(main_env, main_agent))

        # agents = distribute_agents(main_agent, agents)


        if(runs%10==0):
            ###############PLOT##################
            plt.figure(figsize=[12, 9])
            plt.subplot(1, 1, 1)
            plt.title(ENV_NAME)
            plt.xlabel('Instances of Training')
            plt.ylabel('Reward')
            plt.plot(scores_agent, color='green', marker='o',
                    markerfacecolor='blue', label='single_agent')
            plt.plot(scores_main_agent, color='red', marker='o',
                    markerfacecolor='blue', label='aggregated_agent({})'.format(NO_OF_TRAINERS))
            plt.grid()
            plt.legend()

            # plt.show()
            plt.savefig(ENV_NAME+'_'+str(NO_OF_TRAINERS)+'plot.png')
            plt.close()

