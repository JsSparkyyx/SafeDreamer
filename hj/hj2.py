import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = "cpu"  # or "cuda" if available

# 1. Dataset Definition
# Expects NPZ file with:
#   "states": shape (N, 4) --> [car_x, car_y, obs_x, obs_y]
#   "actions": shape (N, 2)
#   "next_states": shape (N, 4)
class SafetyQDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.s = data["s"]["feature"]
        self.s_next = data["s_next"]["feature"]
        self.actions = data["actions"]
        self.costs = data["costs"]
        self.values = data["s"]["value"]
        

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        s = torch.tensor(self.states[idx], dtype=torch.float32)
        a = torch.tensor(self.actions[idx], dtype=torch.float32)
        s_next = torch.tensor(self.next_states[idx], dtype=torch.float32)
        h_s = torch.tensor(self.values[idx], dtype=torch.float32)
        return s, a, s_next, h_s

# 2. Q-Network Definition
# Q: (s, a) --> scalar. Input dim: state (4) concatenated with action (2) = 6.
class QNetwork(nn.Module):
    def __init__(self, input_dim=1536, action_dim=2, hidden_dim=1024, output_dim=1):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 3. Candidate Action Generator
def get_candidate_actions(num_candidates=40, max_velocity=3.0):
    # Increase resolution by setting num_candidates higher.
    left = np.linspace(0, 1, num_candidates, endpoint=False)
    right = np.linspace(0, 1, num_candidates, endpoint=True)
    actions = np.meshgrid(left, right)
    actions = np.stack(actions, axis=-1).reshape(-1, 2)  # shape (num_candidates, 2)
    return torch.tensor(actions, dtype=torch.float32)  # shape (num_candidates, 2)

# 4. Helper: Compute h(s)
def compute_h(s):
    # s: (..., 4) with first two elements = car position, last two = obstacle center.
    car_pos = s[..., :2]
    obs_pos = s[..., 2:4]
    return torch.norm(car_pos - obs_pos, dim=-1, keepdim=True) - 4.0

# 5. Training Loop with Target Network and Gradient Clipping
def train_safety_q(q_net, target_net, dataloader, candidate_actions,
                   gamma=0.99, num_epochs=50, lr=1e-3, target_update_freq=5):
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    q_net.train()
    target_net.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for s, a, s_next, h_s in dataloader:
            optimizer.zero_grad()
            
            # Current Q(s,a)
            sa = torch.cat([s, a], dim=1)  # shape (B,6)
            current_Q = q_net(sa)  # shape (B,1)
            
            # # h(s)
            # h_s = compute_h(s)  # shape (B,1)
            
            # Evaluate Q(s_next, a') over candidate actions using the target network:
            B = s_next.shape[0]
            N_candidates = candidate_actions.shape[0]
            s_next_expanded = s_next.unsqueeze(1).expand(-1, N_candidates, -1)
            candidate_actions_expanded = candidate_actions.unsqueeze(0).expand(B, -1, -1)
            sa_next = torch.cat([s_next_expanded, candidate_actions_expanded], dim=2)
            # sa_next_flat = sa_next.reshape(-1, 6)
            with torch.no_grad():
                Q_next_flat = target_net(sa_next)  # Use target network here
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
            
# 6. Visualization Function with Safe Policy Overlay
# This function visualizes:
# - The learned safety value function V(s) = maxₐ Q(s,a) as a heatmap.
# - The safe policy as a vector field indicating the best candidate action at each grid point.
def visualize_value_function_and_policy(q_net, candidate_actions, device,
                                          obs_center=np.array([10, 11]),
                                          grid_limits=(-30, 30), num_points=100,
                                          quiver_step=2):  # set a smaller step for higher density
    # Create a grid of x,y positions.
    xs = np.linspace(grid_limits[0], grid_limits[1], num_points)
    ys = np.linspace(grid_limits[0], grid_limits[1], num_points)
    X, Y = np.meshgrid(xs, ys)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # shape (num_points^2, 2)
    
    # Construct states: each state is [car_x, car_y, obs_center_x, obs_center_y]
    obs_tile = np.tile(obs_center.reshape(1, 2), (grid_points.shape[0], 1))
    s_grid = np.concatenate([grid_points, obs_tile], axis=1)  # (B, 4)
    s_grid_tensor = torch.tensor(s_grid, dtype=torch.float32, device=device)
    
    B = s_grid_tensor.shape[0]
    N_candidates = candidate_actions.shape[0]
    
    # Expand state to create input for each candidate action.
    s_grid_expanded = s_grid_tensor.unsqueeze(1).expand(-1, N_candidates, -1)
    candidate_actions_expanded = candidate_actions.unsqueeze(0).expand(B, -1, -1)
    sa_grid = torch.cat([s_grid_expanded, candidate_actions_expanded], dim=2)
    sa_grid_flat = sa_grid.reshape(-1, 6)
    
    q_net.eval()
    with torch.no_grad():
        Q_values_flat = q_net(sa_grid_flat)
    Q_values = Q_values_flat.reshape(B, N_candidates)  # shape (B, N_candidates)
    
    # For each state, take the maximum Q value (value function) and get the index of the best candidate action.
    V_values, best_indices = torch.max(Q_values, dim=1)  # V_values shape (B, 1); best_indices shape (B,)
    V_values_grid = V_values.reshape(num_points, num_points).cpu().numpy()
    
    # Extract best candidate actions and reshape to grid.
    safe_actions = candidate_actions[best_indices]  # shape (B, 2)
    safe_actions_grid = safe_actions.reshape(num_points, num_points, 2).cpu().numpy()
    
    # Plot the value function as a heatmap.
    plt.figure(figsize=(10, 8))
    im = plt.imshow(V_values_grid, extent=(grid_limits[0], grid_limits[1],
                                             grid_limits[0], grid_limits[1]),
                    origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(im, label="Safety Value V(s)")
    plt.xlabel("Car X Position")
    plt.ylabel("Car Y Position")
    plt.title("Learned Hamilton–Jacobi Safety Value Function with Safe Policy")
    
    # Overlay the obstacle as a red circle.
    obstacle_circle = plt.Circle((obs_center[0], obs_center[1]), 4.0, color='red',
                                 fill=False, linewidth=2, label="Obstacle")
    plt.gca().add_patch(obstacle_circle)
    
    # Downsample the grid for the vector field (quiver) to avoid clutter.
    X_ds = X[::quiver_step, ::quiver_step]
    Y_ds = Y[::quiver_step, ::quiver_step]
    U = safe_actions_grid[::quiver_step, ::quiver_step, 0]
    V = safe_actions_grid[::quiver_step, ::quiver_step, 1]
    
    # Overlay the safe policy as arrows (vector field).
    # plt.quiver(X_ds, Y_ds, U, V, color='white', alpha=0.5, scale=20, width=0.005, label="Safe Policy")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Load dataset from "safe_rl_dataset.npz"
    dataset = SafetyQDataset("episodes.npz")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize Q-network and target network
    q_net = QNetwork(input_dim=6, hidden_dim=64, output_dim=1).to(device)
    target_net = QNetwork(input_dim=6, hidden_dim=64, output_dim=1).to(device)
    target_net.load_state_dict(q_net.state_dict())  # initial copy
    
    # Generate candidate actions (with increased resolution)
    candidate_actions = get_candidate_actions(num_candidates=40, max_velocity=3.0).to(device)
    
    # Training hyperparameters
    gamma = 0.999999
    num_epochs = 10
    lr = 1e-3
    target_update_freq = 1  # update target network every epoch in this example
    
    print("Starting training of the safety Q-function ...")
    train_safety_q(q_net, target_net, dataloader, candidate_actions,
                   gamma=gamma, num_epochs=num_epochs, lr=lr,
                   target_update_freq=target_update_freq)
    print("Training complete.")
    
    # Visualize the learned safety value function and the derived safe policy.
    visualize_value_function_and_policy(q_net, candidate_actions, device=device,
                               obs_center=np.array([10, 11]),
                               grid_limits=(-30, 30), num_points=100, quiver_step=5)
