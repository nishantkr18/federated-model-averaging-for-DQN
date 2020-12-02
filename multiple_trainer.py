from dqn import *
from agent_compiler import *

def test_agent(env, agent, runs = 5):
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
    print("Avg test score for {} runs: {}".format(runs, sum(avg_score)/len(avg_score)))
    return sum(avg_score)/len(avg_score)


if __name__ == "__main__":
    ENV_NAME = "CartPole-v0"
    # environment
    envs = []
    main_env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    NO_OF_TRAINERS = 2
    for i in range(NO_OF_TRAINERS):
        envs.append(gym.make(ENV_NAME))

    main_agent = DQNAgent(main_env)

    agents = [DQNAgent(envs[i], network=main_agent.dqn) for i in range(NO_OF_TRAINERS)]
    scores=[]
    for runs in range(1000):
        for (i, agent) in enumerate(agents):
            # training each agent serially (needs to be parallelized)
            agent.train(200)
            scores.append(test_agent(test_env, agent))

        # main_agent = combine_agents_reward_based(main_agent, agents, scores)
        main_agent = combine_agents(main_agent, agents)
        print('Testing main_agent')
        
        test_agent(main_env, main_agent)

        # agents = distribute_agents(main_agent, agents)

    env.close()
