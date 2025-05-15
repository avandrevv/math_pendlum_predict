from models import FCNetwork, LNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def train_model(model, X_train, y_train, epochs=100, dt=0.001, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-5)
    # L2-регуляризация^

    losses = []

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(len(X_train)):
            x = torch.tensor(X_train[i], dtype=torch.float32)
            y = torch.tensor(y_train[i], dtype=torch.float32)
            if isinstance(model, LNN):
                q = x[:, :2]
                q_dot = x[:, 2:]
                pred = model.compute_dynamics(q, q_dot)

                predicted_q_dot = q_dot + dt * pred
                predicted_q = q + dt * q_dot

                mse_loss = F.mse_loss(torch.cat([predicted_q, predicted_q_dot], dim=-1), y)
                loss = torch.sqrt(mse_loss)
            else:
                pred = model(x)
                mse_loss = F.mse_loss(pred, y)
                loss = torch.sqrt(mse_loss)

            optimizer.zero_grad() 
            loss.backward()
                        
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step() 
            total_loss += loss.item()
        avg_loss = total_loss / len(X_train)
        losses.append(avg_loss)
        print(f"Эпоха {epoch + 1}/{epochs}, Потери: {avg_loss:.4f}")
    return losses