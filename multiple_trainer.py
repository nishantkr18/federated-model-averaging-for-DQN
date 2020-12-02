from dqn import *
from agent_compiler import *

def test_agent(env, agent, runs = 5):
    avg_score = []
    for i in range(runs):
        agent.is_test = True
        state = env.reset()
        done = False
        score = 0
        while not done:
            # env.render()
            action = agent.select_action(state)
            next_state, reward, done = agent.step(action)

            state = next_state
            score += reward
        avg_score.append(score)
    print("Avg test score for {} runs: {}".format(runs, sum(avg_score)/len(avg_score)))
    return sum(avg_score)/len(avg_score)

if __name__ == "__main__":
    # environment
    env = gym.make("CartPole-v0")

    NO_OF_TRAINERS = 2
    main_agent = DQNAgent(env)

    agents = [DQNAgent(env, network=main_agent.dqn) for i in range(NO_OF_TRAINERS)]
    scores=[]
    for runs in range(100):
        for (i, agent) in enumerate(agents):
            # training each agent serially (needs to be parallelized)
            agent.train(210)
            scores.append(test_agent(env, agent))

        main_agent = combine_agents_reward_based(main_agent, agents, scores)
        print('Testing main_agent')
        
        test_agent(env, main_agent)

        agents = distribute_agents(main_agent, agents)

    env.close()
