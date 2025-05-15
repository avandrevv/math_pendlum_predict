
# data_generation.py
import numpy as np
from scipy.integrate import solve_ivp

class DoublePendulum:
    def __init__(self, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
        self.params = {'m1': m1, 'm2': m2, 'L1': L1, 'L2': L2, 'g': g}
    
    def equations(self, state, t):
        theta1, z1, theta2, z2 = state
        m1, m2, L1, L2, g = self.params.values()
        delta = theta1 - theta2
        c, s = np.cos(delta), np.sin(delta)
        
        denominator = m1 + m2 * s**2
        z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1**2 * c + L2 * z2**2) - (m1 + m2) * g * np.sin(theta1))
        z1dot /= L1 * denominator

        z2dot = (m1 + m2) * (L1 * z1**2 * s - g * np.sin(theta2) + m2 * L2 * z2**2 * s * c)
        z2dot += (m1 + m2) * g * np.sin(theta1) * c
        z2dot /= L2 * denominator

        return [z1, z1dot, z2, z2dot]

    def generate_trajectory(self, initial_state, t):
        def wrapped_equations(t, state):
            return self.equations(state, t)
        
        solution = solve_ivp(wrapped_equations, [t[0], t[-1]], initial_state, t_eval=t, method='RK45')
        return np.hstack([solution.y[:2].T, solution.y[2:].T])
    
    def generate_dataset(self, num_samples=20, time_steps=1500, duration=15, filename="dataset.npz"):
        duration = 15
        desired_dt = 0.001
        time_steps = int(duration / desired_dt) + 1 
        t = np.linspace(0, duration, time_steps)
        dt = t[1] - t[0] 

        X, y = [], []

        for _ in range(num_samples):
            initial_state = np.random.uniform([-np.pi, -1, -np.pi, -1], [np.pi, 1, np.pi, 1])
            trajectory = self.generate_trajectory(initial_state, t)
            X.append(trajectory[:-1]) 
            y.append(trajectory[1:])

        X, y = np.array(X), np.array(y)

        self.save_dataset(X, y, dt, filename)
        print(f"Данные сохранены в файл {filename}")

    @staticmethod
    def save_dataset(X, y,dt, filename):
        """Сохраняет набор данных в файл"""
        np.savez_compressed(filename, X=X, y=y, dt=dt)



if __name__ == "__main__":
    pendulum = DoublePendulum()
    pendulum.generate_dataset(num_samples=20, time_steps=1500, duration=15, filename="dataset.npz")
