from dqn import *
from agent_compiler import combine_agents

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

if __name__ == "__main__":
    # environment
    env = gym.make("CartPole-v0")
    main_agent = DQNAgent(env)

    NO_OF_TRAINERS = 10

    agents = [DQNAgent(env, network=main_agent.dqn) for i in range(NO_OF_TRAINERS)]

    for run in range(100):
        for (i, agent) in enumerate(agents):
            # training 
            agent.train(250)

        main_agent = combine_agents(env, agents)
        print('Testing main_agent')
        test_agent(env, main_agent)


    env.close()
