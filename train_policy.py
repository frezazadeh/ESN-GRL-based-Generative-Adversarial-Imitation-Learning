import os
import gym
import torch
import numpy as np
from torch.distributions import Categorical
from policy_agent.agent import GraphReinforceAgent, Utils, GraphDataProcessor

def main():
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 1234
    Utils.set_seed(seed)
    # Reset the environment with seed (Gym v0.26+ returns a tuple: (obs, info))
    result = env.reset(seed=seed)
    state = result[0] if isinstance(result, tuple) else result
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    learning_rate = 0.0005
    episodes = 500
    gamma = 0.99
    print_interval = 10

    agent = GraphReinforceAgent(
        input_dimension=state_dim,
        output_dimension=action_dim,
        esn_reservoir_size=500,
        hidden_layer_dimension=128,
        learning_rate=learning_rate
    ).to(device)
    score_list = []
    index_pairs = GraphDataProcessor.construct_index_pairs(state_dim)

    for episode in range(episodes):
        # Reset environment for each episode using the new API
        result = env.reset(seed=seed)
        state = result[0] if isinstance(result, tuple) else result

        score = 0
        done = False

        while not done:
            # Convert state to tensor and compute the ESN state
            state_tensor = torch.tensor(state, dtype=torch.float).to(device)
            esn_state = agent.esn(state_tensor)
            # Create graph data from the raw state
            graph_data = GraphDataProcessor.create_graph_data(state, index_pairs)
            # Forward pass through the agent's network to get action probabilities
            action_probs = agent(
                graph_data.x.to(device),
                graph_data.edge_index.to(device),
                esn_state.unsqueeze(0)
            )
            # Sample an action from the categorical distribution
            action_distribution = Categorical(torch.exp(action_probs))
            action = action_distribution.sample()
            log_prob = action_probs[0, action]
            
            # Gym's step() may return 5 values (obs, reward, terminated, truncated, info)
            step_result = env.step(action.item())
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            agent.store_transition((reward, log_prob))
            state = next_state
            score += reward

        agent.optimize(gamma)
        score_list.append(score)

        if (episode + 1) % print_interval == 0:
            avg_score = sum(score_list[-print_interval:]) / print_interval
            print(f"Episode {episode+1}, Avg Score: {avg_score}")

    # Save the trained policy checkpoint with the .ckpt extension
    experts_dir = os.path.join("experts", "CartPole-v1")
    if not os.path.exists(experts_dir):
        os.makedirs(experts_dir)
    torch.save(agent.state_dict(), os.path.join(experts_dir, "policy.ckpt"))
    
    env.close()

if __name__ == "__main__":
    main()
