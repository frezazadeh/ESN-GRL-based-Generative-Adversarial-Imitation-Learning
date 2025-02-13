import numpy as np
import torch
from torch.nn import Module

from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch

# Remove deprecated default tensor type setting.
# Instead, we will always create tensors with torch.tensor(..., dtype=torch.float, device=device).

# Helper function to process Gym observations.
def process_obs(obs):
    # Gym v0.26+ returns (observation, info); extract the observation.
    if isinstance(obs, tuple):
        return obs[0]
    return obs

class GAIL(Module):
    def __init__(self, state_dim, action_dim, discrete, train_config=None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)
        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()
        # Ensure the state is a tensor on the proper device.
        device = next(self.parameters()).device
        state = torch.tensor(state, dtype=torch.float, device=device)
        distb = self.pi(state)
        action = distb.sample().detach().cpu().numpy()
        return action

    def train(self, env, expert, render=False):
        # Determine the device from the model's parameters.
        device = next(self.parameters()).device

        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        opt_d = torch.optim.Adam(self.d.parameters())

        exp_rwd_iter = []
        exp_obs = []
        exp_acts = []

        steps = 0
        # Expert demonstration loop.
        while steps < num_steps_per_iter:
            ep_obs = []
            ep_rwds = []
            t = 0
            done = False

            ob = process_obs(env.reset())
            while not done and steps < num_steps_per_iter:
                act = expert.act(ob)
                ep_obs.append(ob)
                exp_obs.append(ob)
                exp_acts.append(act)

                if render:
                    env.render()
                # Handle step() return (4 or 5 values).
                step_result = env.step(act)
                if len(step_result) == 5:
                    ob, rwd, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    ob, rwd, done, info = step_result
                ob = process_obs(ob)
                ep_rwds.append(rwd)
                t += 1
                steps += 1
                if horizon is not None and t >= horizon:
                    done = True
                    break
            if done:
                exp_rwd_iter.append(np.sum(ep_rwds))

        # Convert expert observations and actions to tensors.
        try:
            exp_obs = torch.tensor(np.array(exp_obs), dtype=torch.float, device=device)
        except Exception as e:
            print("Error converting expert observations to tensor:", e)
            raise e
        exp_acts = torch.tensor(np.array(exp_acts), dtype=torch.float, device=device)

        exp_rwd_mean = np.mean(exp_rwd_iter)
        print("Expert Reward Mean: {}".format(exp_rwd_mean))

        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []
            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < num_steps_per_iter:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_gms = []
                ep_lmbs = []
                t = 0
                done = False

                ob = process_obs(env.reset())
                while not done and steps < num_steps_per_iter:
                    act = self.act(ob)
                    ep_obs.append(ob)
                    obs.append(ob)
                    ep_acts.append(act)
                    acts.append(act)
                    if render:
                        env.render()
                    step_result = env.step(act)
                    if len(step_result) == 5:
                        ob, rwd, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        ob, rwd, done, info = step_result
                    ob = process_obs(ob)
                    ep_rwds.append(rwd)
                    ep_gms.append(gae_gamma ** t)
                    ep_lmbs.append(gae_lambda ** t)
                    t += 1
                    steps += 1
                    if horizon is not None and t >= horizon:
                        done = True
                        break
                if done:
                    rwd_iter.append(np.sum(ep_rwds))
                ep_obs = torch.tensor(np.array(ep_obs), dtype=torch.float, device=device)
                ep_acts = torch.tensor(np.array(ep_acts), dtype=torch.float, device=device)
                ep_rwds = torch.tensor(ep_rwds, dtype=torch.float, device=device)
                ep_gms = torch.tensor(ep_gms, dtype=torch.float, device=device)
                ep_lmbs = torch.tensor(ep_lmbs, dtype=torch.float, device=device)

                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs
                ep_disc_rets = torch.tensor([sum(ep_disc_costs[i:]) for i in range(t)],
                                             dtype=torch.float, device=device)
                ep_rets = ep_disc_rets / ep_gms
                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat((self.v(ep_obs)[1:], torch.tensor([[0.]], dtype=torch.float, device=device))).detach()
                ep_deltas = ep_costs.unsqueeze(-1) + gae_gamma * next_vals - curr_vals
                ep_advs = torch.tensor([((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:]).sum() for j in range(t)],
                                         dtype=torch.float, device=device)
                advs.append(ep_advs)
                gms.append(ep_gms)

            rwd_iter_means.append(np.mean(rwd_iter))
            print("Iterations: {},   Reward Mean: {}".format(i + 1, np.mean(rwd_iter)))

            obs = torch.tensor(np.array(obs), dtype=torch.float, device=device)
            acts = torch.tensor(np.array(acts), dtype=torch.float, device=device)
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)
            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)
            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(exp_scores, torch.zeros_like(exp_scores)) + \
                   torch.nn.functional.binary_cross_entropy_with_logits(nov_scores, torch.ones_like(nov_scores))
            loss.backward()
            opt_d.step()

            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()
            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()
            grad_diff = get_flat_grads(constraint(), self.v)
            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v).detach()
                return hessian
            g = get_flat_grads(((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v).detach()
            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))
            new_params = old_params + alpha * s
            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)
            def L():
                distb = self.pi(obs)
                return (advs * torch.exp(distb.log_prob(acts) - old_distb.log_prob(acts).detach())).mean()
            def kld():
                distb = self.pi(obs)
                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs
                    return (old_p * (torch.log(old_p) - torch.log(p))).sum(-1).mean()
                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)
                    return (0.5) * (((old_cov / cov).sum(-1) + (((old_mean - mean) ** 2) / cov).sum(-1) - self.action_dim + torch.log(cov).sum(-1) - torch.log(old_cov).sum(-1))).mean()
            grad_kld_old_param = get_flat_grads(kld(), self.pi)
            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_kld_old_param, v), self.pi).detach()
                return hessian + cg_damping * v
            g = get_flat_grads(L(), self.pi).detach()
            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()
            new_params = rescale_and_linesearch(g, s, Hs, max_kl, L, kld, old_params, self.pi)
            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts)).mean()
            grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy, self.pi)
            new_params += lambda_ * grad_disc_causal_entropy
            set_params(self.pi, new_params)

        return exp_rwd_mean, rwd_iter_means
