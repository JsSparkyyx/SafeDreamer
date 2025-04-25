import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import load_agent
import pickle
from tqdm import tqdm
from formularone import SafeGymnasiumEval
import cv2

device = "cuda"  # or "cuda" if available

# 1. Dataset Definition
# Expects NPZ file with:
#   "states": shape (N, 4) --> [car_x, car_y, obs_x, obs_y]
#   "actions": shape (N, 2)
#   "next_states": shape (N, 4)
class SafetyQDataset(Dataset):
    def __init__(self, npz_file):
        with open(npz_file, "rb") as f:
            s, actions, costs = pickle.load(f)
        self.latents = s["latent"]
        self.embs = s["emb"]
        self.values = s["value"]
        

    def __len__(self):
        return self.embs.shape[0]

    def __getitem__(self, idx):
        latent = {k:torch.tensor(self.latents[k][idx], dtype=torch.float32) for k in self.latents.keys()}
        embs = torch.tensor(self.embs[idx], dtype=torch.float32)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return latent, embs, value

# 2. Q-Network Definition
# Q: (s, a) --> scalar. Input dim: state (4) concatenated with action (2) = 6.
class QNetwork(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, output_dim=1):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 3. Candidate Action Generator
def get_candidate_actions(num_candidates=40):
    # Increase resolution by setting num_candidates higher.
    left = np.linspace(0, 1, num_candidates, endpoint=False)
    right = np.linspace(0, 1, num_candidates, endpoint=True)
    actions = np.meshgrid(left, right)
    actions = np.stack(actions, axis=-1).reshape(-1, 2)  # shape (num_candidates, 2)
    return torch.tensor(actions, dtype=torch.float32).to(device)  # shape (num_candidates, 2)

if __name__ == "__main__":
    # Set seeds for reproducibility
    agent, agent_reach = load_agent()
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Load dataset from "safe_rl_dataset.npz"
    ckpt = torch.load("hj100.pt")
    
    # Initialize Q-network and target network
    q_net = QNetwork().to(device)
    q_net.load_state_dict(ckpt)
    
    # Generate candidate actions (with increased resolution)
    candidate_actions = get_candidate_actions(num_candidates=20).to(device)
    train_envs = SafeGymnasiumEval(name="SafetyCarGoal2Vision-v0")
    agent_state = None
    agent_state_reach = None
    threshold = 0.2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4
    videoWriter = cv2.VideoWriter(f"test.mp4",fourcc,20,(128,128))
    obs = train_envs.reset()
    obs = {k: np.expand_dims(np.array(obs[k]),axis=0) for k in obs}
    videoWriter.write(obs['image'][0])
    action, agent_state, embed = agent(obs, agent_state)
    action_reach, agent_state_reach, embed = agent_reach(obs, agent_state_reach)
    (latent, _) = agent_state
    feat = agent._wm.dynamics.get_feat(latent)
    value = q_net(feat).flatten()[0].detach().cpu().numpy()
    if value > -threshold:
        action = action_reach
        agent_state = (latent, action["action"])
    else:
        agent_state_reach = (agent_state_reach[0], action["action"])
    action = action["action"].cpu().numpy()
    for i in range(1000):
        obs, reward, done, info = train_envs.step(action[0])
        obs = {k: np.expand_dims(np.array(obs[k]),axis=0) for k in obs}
        action, agent_state, embed = agent(obs, agent_state)
        action_reach, agent_state_reach, embed = agent_reach(obs, agent_state_reach)
        (latent, _) = agent_state
        feat = agent._wm.dynamics.get_feat(latent)
        value = q_net(feat).flatten()[0].detach().cpu().numpy()
        if value > -threshold:
            action = action_reach
            agent_state = (latent, action["action"])
        else:
            agent_state_reach = (agent_state_reach[0], action["action"])
        # action = action_reach
        action = action["action"].cpu().numpy()
        videoWriter.write(obs['image'][0])
    videoWriter.release()
