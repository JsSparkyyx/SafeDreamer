import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
sys.path.append("dreamerv3-torch")
from torch.utils.data import Dataset, DataLoader
from utils import load_agent
import pickle
from tqdm import tqdm
from formularone import SafeGymnasiumEval
import cv2

device = "cuda"  # or "cuda" if available

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

def get_candidate_actions(num_candidates=100):
    left = np.linspace(-1, 1, num_candidates, endpoint=False)
    right = np.linspace(-1, 1, num_candidates, endpoint=True)
    actions = np.meshgrid(left, right)
    actions = np.stack(actions, axis=-1).reshape(-1, 2)
    return torch.tensor(actions, dtype=torch.float32).to(device)

if __name__ == "__main__":
    agent, agent_reach = load_agent()
    candidate_actions = get_candidate_actions(num_candidates=20).to(device)
    torch.manual_seed(0)
    np.random.seed(0)
    
    ckpt = torch.load("hj100_old.pt")
    q_net = QNetwork().to(device)
    q_net.load_state_dict(ckpt)
    
    train_envs = SafeGymnasiumEval(name="SafetyCarGoal2Vision-v0")
    agent_state = None
    agent_state_reach = None
    threshold = 0.4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(f"test_real.mp4", fourcc, 20, (128, 128))

    obs = train_envs.reset()
    obs = {k: np.expand_dims(np.array(obs[k]), axis=0) for k in obs}

    latent = action = None
    action, agent_state, embed = agent(obs, agent_state)
    action_reach, agent_state_reach, embed_reach = agent_reach(obs, agent_state_reach)
    (latent_cur, _) = agent_state
    feat_cur = agent._wm.dynamics.get_feat(latent_cur)
    value = q_net(feat_cur).flatten()[0].detach().cpu().numpy()
    if value > -threshold:
        action = action_reach["action"].cpu().numpy()
        agent_state = (latent_cur, action_reach["action"])
    else:
        latent = {k:l.expand((candidate_actions.shape[0],)+l.shape[1:]) for k,l in latent_cur.items()}
        embed = embed.expand((candidate_actions.shape[0],) + embed.shape[1:])
        latent, _ = agent._wm.dynamics.obs_step(latent, candidate_actions, embed, torch.ones((candidate_actions.shape[0], 1)).to(device))
        feat = agent._wm.dynamics.get_feat(latent)
        value = q_net(feat).view(-1,1)
        idx = torch.argmax(value)
        action = candidate_actions[idx].unsqueeze(0).cpu().numpy()
        agent_state = (latent_cur, candidate_actions[idx].unsqueeze(0))
        agent_state_reach = (agent_state_reach[0], candidate_actions[idx].unsqueeze(0))

    for i in range(1000):
        obs, reward, done, info = train_envs.step(action[0])
        obs = {k: np.expand_dims(np.array(obs[k]), axis=0) for k in obs}
        action, agent_state, embed = agent(obs, agent_state)
        action_reach, agent_state_reach, embed_reach = agent_reach(obs, agent_state_reach)
        (latent_cur, _) = agent_state
        feat_cur = agent._wm.dynamics.get_feat(latent_cur)
        value = q_net(feat_cur).flatten()[0].detach().cpu().numpy()
        if value > -threshold:
            action = action_reach["action"].cpu().numpy()
            agent_state = (latent_cur, action_reach["action"])
        else:
            latent = {k:l.expand((candidate_actions.shape[0],)+l.shape[1:]) for k,l in latent_cur.items()}
            embed = embed.expand((candidate_actions.shape[0],) + embed.shape[1:])
            latent, _ = agent._wm.dynamics.obs_step(latent, candidate_actions, embed, torch.zeros((candidate_actions.shape[0], 1)).to(device))
            feat = agent._wm.dynamics.get_feat(latent)
            value = q_net(feat).view(-1,1)
            idxs = torch.argsort(value, descending=True)
            action = candidate_actions[idxs][:3000]
            nom_action = action_reach["action"][0]
            dis = torch.norm(action - nom_action, dim=1)
            idx = torch.argmin(dis)
            agent_state = (latent_cur, action[idx])
            agent_state_reach = (agent_state_reach[0], action[idx])
            action = action[idx].cpu().numpy()

        # ----- Overlay HJ value and display -----
        frame = obs['image'][0].copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Handle both scalar and batched value cases
        if torch.is_tensor(value) and value.numel() > 1:
            hj_val = float(value.max().detach().cpu().numpy())  # or mean(), or some other stat
        elif torch.is_tensor(value):
            hj_val = float(value.detach().cpu().numpy())
        else:
            hj_val = value

        cv2.putText(frame, f"HJ: {hj_val:.3f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
        display_frame = cv2.resize(frame, (512, 512))  # or any larger size
        cv2.imshow("Simulation", display_frame)
        cv2.waitKey(1)
        videoWriter.write(frame)

    videoWriter.release()
    cv2.destroyAllWindows()
