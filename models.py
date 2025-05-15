
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            # nn.Softplus(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)
    
    


class LNN(nn.Module):
 
    def __init__(self, input_size, hidden_size):
        super(LNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            # nn.Softplus(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            # nn.Dropout(p=0.1),
            nn.Linear(hidden_size, 1) 
)


    def forward(self, q, q_dot):
        x = torch.cat([q, q_dot], dim=-1)
        lagrangian = self.net(x)
        return lagrangian
    
    
    def compute_dynamics(self, q, q_dot):
        q.requires_grad_(True)
        q_dot.requires_grad_(True)

        L = self.forward(q, q_dot)

        dL_dq = torch.autograd.grad(L.sum(), q, create_graph=True)[0]
        dL_dqdot = torch.autograd.grad(L.sum(), q_dot, create_graph=True)[0]

        batch_size = q.size(0)
        d2L_dqdot2 = torch.zeros((batch_size, 2, 2), device=q.device)
        d2L_dqdot_dq = torch.zeros((batch_size, 2, 2), device=q.device)

        for i in range(2):
            d2L_dqdot2[:, i, :] = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q_dot, retain_graph=True
            )[0]
            d2L_dqdot_dq[:, i, :] = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q, retain_graph=True
            )[0]

        eps = 1e-6 * torch.eye(2, device=q.device).unsqueeze(0)
        d2L_dqdot2 = d2L_dqdot2 + eps

        q_dot_ = q_dot.unsqueeze(-1) 
        right_hand_side = dL_dq.unsqueeze(-1) - torch.bmm(d2L_dqdot_dq, q_dot_)

        try:
            q_ddot = torch.linalg.solve(d2L_dqdot2, right_hand_side) 
        except RuntimeError:
            q_ddot = torch.linalg.pinv(d2L_dqdot2) @ right_hand_side

        return q_ddot.squeeze(-1)
