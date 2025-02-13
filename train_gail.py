import os
import json
import pickle
import argparse

import torch
import gym

from models.gail import GAIL
from policy_agent.agent import GraphReinforceAgent, Utils, GraphDataProcessor

# -----------------------------------------------------------------------------
# GRLExpert wrapper so the GRL policy (trained via train_policy.py) can be used as the expert.
# -----------------------------------------------------------------------------
class GRLExpert:
    def __init__(self, state_dim, action_dim, device, checkpoint_path, seed=1234):
        self.device = device
        # Instantiate the GRL agent with the same configuration as in train_policy.py.
        self.agent = GraphReinforceAgent(
            input_dimension=state_dim,
            output_dimension=action_dim,
            esn_reservoir_size=500,
            hidden_layer_dimension=128,
            learning_rate=0.0005
        ).to(device)
        # Load the GRL agent checkpoint (saved as a .ckpt file).
        self.agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.agent.eval()
        # Precompute the index pairs used to construct graph data.
        self.index_pairs = GraphDataProcessor.construct_index_pairs(state_dim)

    def act(self, state):
        # If state is a tuple (observation, info), extract the observation.
        if isinstance(state, tuple):
            state = state[0]
        # Ensure that state is not empty.
        if len(state) == 0:
            raise ValueError("Received an empty state.")
        # Convert state to a tensor.
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        # Compute the Echo State Network (ESN) output.
        esn_state = self.agent.esn(state_tensor)
        # Construct graph data from the state.
        graph_data = GraphDataProcessor.create_graph_data(state, self.index_pairs)
        # Forward pass through the GRL agent network to get action probabilities.
        action_probs = self.agent(
            graph_data.x.to(self.device),
            graph_data.edge_index.to(self.device),
            esn_state.unsqueeze(0)
        )
        # Sample an action from the categorical distribution.
        from torch.distributions import Categorical
        action_distribution = Categorical(torch.exp(action_probs))
        action = action_distribution.sample()
        return action.item()

# -----------------------------------------------------------------------------
# Main GAIL training function.
# -----------------------------------------------------------------------------
def main():
    env_name = "CartPole-v1"

    # Create the checkpoint folder if it doesn't exist.
    ckpt_dir = "ckpts"
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    # Expert checkpoint path (saved by train_policy.py).
    expert_ckpt_path = os.path.join("experts", env_name, "policy.ckpt")

    # Load the imitation (GAIL) training configuration.
    with open("config.json", "r") as f:
        config = json.load(f)[env_name]
    # Save the config to the checkpoint folder for record.
    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(env_name)
    # Reset the environment once to determine state dimensions.
    result = env.reset()
    # For CartPole-v1, state is a vector of length 4.
    state_dim = env.observation_space.shape[0]
    discrete = True
    action_dim = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 1234
    Utils.set_seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Wrap the GRL policy as the expert.
    expert = GRLExpert(state_dim, action_dim, device, expert_ckpt_path, seed)

    # Create the GAIL model.
    model = GAIL(state_dim, action_dim, discrete, config).to(device)

    # Train GAIL using the expert's actions.
    results = model.train(env, expert, render=False)
    env.close()

    # Save training results and GAIL checkpoints.
    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    if hasattr(model, "pi"):
        torch.save(model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt"))
    if hasattr(model, "v"):
        torch.save(model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt"))
    if hasattr(model, "d"):
        torch.save(model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
