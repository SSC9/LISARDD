import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(3 * latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, 3 * latent_dim)
        self.log_std = nn.Parameter(torch.zeros(3 * latent_dim))

    def forward(self, z):
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        mu = self.actor_head(x)
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(3 * latent_dim, hidden_dim)
        self.b1 = nn.BatchNorm1d(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.b2 = nn.BatchNorm1d(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.b3 = nn.BatchNorm1d(hidden_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        x = F.relu(self.b1(self.l1(z)))
        x = F.relu(self.b2(self.l2(x)))
        x = F.relu(self.b3(self.l3(x)))
        return self.critic_head(x)


class ActorReinforce(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(3 * latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, 3 * latent_dim)
        self.log_std = nn.Parameter(torch.zeros(3 * latent_dim))

    def forward(self, z):
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        mu = self.actor_head(x)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, batch_size):
        device = next(self.parameters()).device
        rd_seed = torch.randn(batch_size, 3 * self.latent_dim, device=device)
        mu, std = self.forward(rd_seed)
        dist = torch.distributions.Normal(mu, std)
        z = dist.sample()
        log_probs = dist.log_prob(z).sum(dim=1)
        return z, log_probs
