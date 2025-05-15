import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data_generation import DoublePendulum
from main import predict_trajectory
import torch
from models import FCNetwork, LNN

def compute_cartesian_coords(traj, L1=1.0, L2=1.0):
    """Преобразует углы маятников в декартовы координаты"""
    theta1, theta2 = traj[:, 0], traj[:, 2]
    
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    return x1, y1, x2, y2

def animate_pendulum(true_traj, pred_traj, t):
    """Анимация истинной (синей) и предсказанной (красной) траекторий двойного маятника"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Анимация двойного маятника")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid()

    x1_true, y1_true, x2_true, y2_true = compute_cartesian_coords(true_traj)
    x1_pred, y1_pred, x2_pred, y2_pred = compute_cartesian_coords(pred_traj)

    line_true, = ax.plot([], [], 'o-', lw=2, color='blue', alpha=0.6, label="Истинная траектория")
    line_pred, = ax.plot([], [], 'o-', lw=2, color='red', alpha=0.8, linestyle="dashed", label="Предсказанная траектория")
    ax.legend()

    def update(frame):
        """Обновление кадров анимации"""
        line_true.set_data([0, x1_true[frame], x2_true[frame]], [0, y1_true[frame], y2_true[frame]])
        line_pred.set_data([0, x1_pred[frame], x2_pred[frame]], [0, y1_pred[frame], y2_pred[frame]])
        return line_true, line_pred

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=5, blit=True)
    plt.show()

if __name__ == "__main__":
    pendulum = DoublePendulum()
    t = np.linspace(0, 15, 1500)
    initial_state = np.random.uniform([-np.pi, -1, -np.pi, -1], [np.pi, 1, np.pi, 1])
    
    true_trajectory = pendulum.generate_trajectory(initial_state, t)

    # fc = FCNetwork(input_size=4, hidden_size=64, output_size=4)
    # fc.load_state_dict(torch.load("fc_model.pt"))
    lnn = LNN(input_size=4, hidden_size=64)
    lnn.load_state_dict(torch.load("lnn_model.pt"))
    
    pred_trajectory_lnn = predict_trajectory(lnn, initial_state, t)
    # pred_trajectory_lnn = predict_trajectory(fc, initial_state, t)
    
    animate_pendulum(true_trajectory, pred_trajectory_lnn, t)
