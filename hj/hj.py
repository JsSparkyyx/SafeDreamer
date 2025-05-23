import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import load_agent
import pickle
from tqdm import tqdm
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
        self.costs = costs
        

    def __len__(self):
        return self.embs.shape[0]

    def __getitem__(self, idx):
        latent = {k:torch.tensor(self.latents[k][idx], dtype=torch.float32) for k in self.latents.keys()}
        embs = torch.tensor(self.embs[idx], dtype=torch.float32)
        # value = torch.tensor(self.values[idx], dtype=torch.float32)
        value = torch.tensor(self.costs[idx], dtype=torch.float32)
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
    left = np.linspace(-1, 1, num_candidates, endpoint=False)
    right = np.linspace(-1, 1, num_candidates, endpoint=True)
    actions = np.meshgrid(left, right)
    actions = np.stack(actions, axis=-1).reshape(-1, 2)  # shape (num_candidates, 2)
    return torch.tensor(actions, dtype=torch.float32).to(device)  # shape (num_candidates, 2)

# 5. Training Loop with Target Network and Gradient Clipping
def train_safety_q(q_net, target_net, dataloader, candidate_actions, agent,
                   gamma=0.99, num_epochs=50, lr=1e-3, target_update_freq=5):
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    q_net.train()
    target_net.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        torch.save(q_net.state_dict(), "hj.pt")
        for l, embs, h_s in tqdm(dataloader):
            embs = embs.to(device)
            l = {k: v.to(device) for k, v in l.items()}
            h_s = h_s.to(device)
            optimizer.zero_grad()
            
            # Current Q(s,a)
            s = agent._wm.dynamics.get_feat(l)
            current_Q = q_net(s)  # shape (B,1)
            
            # # h(s)
            # h_s = compute_h(s)  # shape (B,1)
            
            # Evaluate Q(s_next, a') over candidate actions using the target network:
            B = s.shape[0]
            N_candidates = candidate_actions.shape[0]
            l_next = {}
            for k, v in l.items():
                shape = [1]*len(v.shape)
                shape[0] = N_candidates
                l_next[k] = v.repeat(*shape)
            a = candidate_actions.unsqueeze(0).expand(B, -1, -1).reshape(B*N_candidates, -1)
            embs = embs.unsqueeze(0).expand(N_candidates, -1, -1).reshape(B*N_candidates, -1)
            is_first = torch.zeros(B*N_candidates, dtype=torch.bool, device=device)
            l_next, _ = agent._wm.dynamics.obs_step(l_next, a, embs, is_first)
            s_next = agent._wm.dynamics.get_feat(l_next)
            # sa_next_flat = sa_next.reshape(-1, 6)
            with torch.no_grad():
                Q_next_flat = target_net(s_next)  # Use target network here
            Q_next = Q_next_flat.reshape(B, N_candidates)
            max_Q_next, _ = torch.max(Q_next, dim=1, keepdim=True)  # (B, 1)
            
            # Target per discounted safety Bellman update:
            min_term = torch.min(h_s, max_Q_next)
            target = (1 - gamma) * h_s + gamma * min_term
            
            loss = criterion(current_Q, target.detach())
            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * B
        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        # Update the target network periodically
        if (epoch + 1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())
            print("Target network updated.")



if __name__ == "__main__":
    # Set seeds for reproducibility
    agent, _ = load_agent()
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Load dataset from "safe_rl_dataset.npz"
    dataset = SafetyQDataset("episodes100.pkl")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Initialize Q-network and target network
    q_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(q_net.state_dict())  # initial copy
    
    # Generate candidate actions (with increased resolution)
    candidate_actions = get_candidate_actions(num_candidates=20).to(device)
    
    # Training hyperparameters
    gamma = 0.999999
    num_epochs = 20
    lr = 1e-3
    target_update_freq = 1  # update target network every epoch in this example
    
    print("Starting training of the safety Q-function ...")
    train_safety_q(q_net, target_net, dataloader, candidate_actions, agent,
                   gamma=gamma, num_epochs=num_epochs, lr=lr,
                   target_update_freq=target_update_freq)
    print("Training complete.")
    # Visualize the learned safety value function and the derived safe policy.
    # visualize_value_function_and_policy(q_net, candidate_actions, device=device,
    #                            obs_center=np.array([10, 11]),
    #                            grid_limits=(-30, 30), num_points=100, quiver_step=5)
