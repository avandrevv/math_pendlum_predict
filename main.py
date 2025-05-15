# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_generation import DoublePendulum
from models import FCNetwork, LNN
from train import train_model

fc = None
lnn = None
fc_loss = None
lnn_loss = None
X = None
y = None
dt = None

def load_dataset(filename="dataset.npz"):
    """Загружает набор данных из файла"""
    try:
        data = np.load(filename)
        print(f"Данные загружены из файла {filename}")
        return data['X'], data['y'], data['dt']
    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
        return None, None, None

def compute_rmse(predictions, true_values):
    """Вычисляет RMSE для траекторий"""
    return np.sqrt(np.mean((predictions - true_values) ** 2))

def predict_trajectory(model, initial_state, t, dt=0.001):
    """
    Предсказывает траекторию модели с использованием метода Рунге–Кутта 4-го порядка.

    Для моделей с методом compute_dynamics (например, LNN):
        - состояние s = [q, q_dot] (4 элемента);
        - вычисляется s_next = [q_dot, q_ddot] через model.compute_dynamics.
    Для моделей без compute_dynamics (например, FCNetwork):
        напрямую предсказывается следующее состояние.
    """
    trajectory = []
    current_state = initial_state.copy()

    for _ in t:
        if hasattr(model, 'compute_dynamics'):
            def f(state):
                q = state[:2]
                q_dot = state[2:]
                q_tensor = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
                q_dot_tensor = torch.tensor(q_dot, dtype=torch.float32).unsqueeze(0)
                q_ddot_tensor = model.compute_dynamics(q_tensor, q_dot_tensor).squeeze(0)
                q_ddot = q_ddot_tensor.detach().numpy().squeeze()
                return np.concatenate([q_dot, q_ddot])
            
            k1 = f(current_state)
            k2 = f(current_state + 0.5 * dt * k1)
            k3 = f(current_state + 0.5 * dt * k2)
            k4 = f(current_state + dt * k3)
            next_state = current_state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            pred_state = next_state
        else:
            pred = model(torch.tensor(current_state, dtype=torch.float32).unsqueeze(0))
            pred_state = pred.squeeze(0).detach().numpy()
        
        trajectory.append(pred_state)
        current_state = pred_state

    return np.array(trajectory)

def plot_combined_phase_portrait(true_traj, pred_traj, model_name):
    """
    Строит единый фазовый портрет для двойного маятника с наложением:
      - Истинная траектория (синие, полупрозрачные)
      - Предсказанная траектория (красные пунктирные)
    
    Для удобства объединяем 4-мерное состояние [q1, q1_dot, q2, q2_dot] в 2D с помощью PCA.
    """
    combined = np.vstack((true_traj, pred_traj))
    mean = np.mean(combined, axis=0)
    centered = combined - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_matrix = Vt.T[:, :2]
    
    true_proj = (true_traj - mean).dot(proj_matrix)
    pred_proj = (pred_traj - mean).dot(proj_matrix)
    
    plt.figure(figsize=(8, 6))
    plt.plot(true_proj[:, 0], true_proj[:, 1], 'b-', alpha=0.5, label="Истинная траектория")
    plt.plot(pred_proj[:, 0], pred_proj[:, 1], 'r--', label="Предсказанная траектория")
    plt.xlabel("Главная компонента 1")
    plt.ylabel("Главная компонента 2")
    plt.title(f"Фазовый портрет двойного маятника ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_save_models():
    """
    Обучает модели (FC и LNN) на загруженных данных и сохраняет их в файл.
    """
    global fc, lnn, fc_loss, lnn_loss, X, y, dt
    if X is None or y is None:
        print("Сначала загрузите данные (пункт 1)!")
        return
    print("\nОбучение полносвязной сети (FC)...")
    fc = FCNetwork(input_size=4, hidden_size=64, output_size=4)
    fc_loss = train_model(fc, X, y, epochs=10)
    print("Обучение полносвязной сети завершено!")
    
    print("\nОбучение лагранжевой сети (LNN)...")
    lnn = LNN(input_size=4, hidden_size=64)
    lnn_loss = train_model(lnn, X, y, epochs=100, dt=dt)
    print("Обучение лагранжевой сети завершено!")
    
    torch.save(fc.state_dict(), "fc_model.pt")
    torch.save(lnn.state_dict(), "lnn_model.pt")
    print("Модели сохранены в 'fc_model.pt' и 'lnn_model.pt'.")

def load_models():
    """
    Загружает сохранённые модели из файлов.
    """
    global fc, lnn
    fc = FCNetwork(input_size=4, hidden_size=64, output_size=4)
    lnn = LNN(input_size=4, hidden_size=64)
    
    if os.path.exists("fc_model.pt") and os.path.exists("lnn_model.pt"):
        fc.load_state_dict(torch.load("fc_model.pt"))
        lnn.load_state_dict(torch.load("lnn_model.pt"))
        print("Сохранённые модели успешно загружены!")
    else:
        print("Файлы моделей не найдены. Сначала обучите модели (опция 2.1).")

def main():
    global X, y, dt, fc, lnn, fc_loss, lnn_loss
    while True:
        print("\nГлавное меню:")
        print("1. Загрузить данные")
        print("2. Управление моделями")
        print("3. Оценка модели через RMSE")
        print("4. Визуализация обучения")
        print("5. Визуализация комбинированного фазового портрета (overlay True vs Predicted)")
        print("6. Временная отладка")
        print("7. Выход")
        choice = input("Выберите действие (1-7): ").strip()
        
        if choice == "1":
            print("\nЗагрузка данных...")
            X, y, dt = load_dataset("dataset.npz")
            if X is not None and y is not None:
                dt = float(dt)
                print("Данные успешно загружены!")
        
        elif choice == "2":
            print("\nУправление моделями:")
            print("  1. Обучить новые модели и сохранить их")
            print("  2. Использовать сохранённые модели")
            print("  0. Вернуться в главное меню")
            sub_choice = input("Выберите действие (0-2): ").strip()
            if sub_choice == "1":
                train_and_save_models()
            elif sub_choice == "2":
                load_models()
            elif sub_choice == "0":
                continue
            else:
                print("Неверный выбор. Попробуйте снова.")
        
        elif choice == "3":
            if fc is None or lnn is None:
                print("Сначала загрузите или обучите модели (пункт 2)!")
            else:
                print("\nОценка модели через RMSE...")
                pendulum = DoublePendulum()
                t = np.linspace(0, 15, 1500)
                initial_state = np.random.uniform(
                    [-np.pi, -1, -np.pi, -1],
                    [np.pi, 1, np.pi, 1]
                )
                true_trajectory = pendulum.generate_trajectory(initial_state, t)
                fc_prediction = predict_trajectory(fc, initial_state, t)
                lnn_prediction = predict_trajectory(lnn, initial_state, t)
                fc_rmse = compute_rmse(fc_prediction, true_trajectory)
                lnn_rmse = compute_rmse(lnn_prediction, true_trajectory)
                print(f"FC RMSE: {fc_rmse}")
                print(f"LNN RMSE: {lnn_rmse}")
        
        elif choice == "4":
            print("\nВизуализация обучения...")
            if fc_loss is not None and lnn_loss is not None:
                plt.plot(fc_loss, label='FC Network')
                plt.plot(lnn_loss, label='LNN')
                plt.xlabel('Эпоха')
                plt.ylabel('Ошибка')
                plt.legend()
                plt.show()
            else:
                print("Сначала выполните обучение моделей (пункт 2)!")
        
        elif choice == "5":
            if fc is None or lnn is None:
                print("Сначала загрузите или обучите модели (пункт 2)!")
                continue
            print("\nВыберите модель для визуализации комбинированного фазового портрета:")
            print("  1. FC")
            print("  2. LNN")
            print("  0. Вернуться в главное меню")
            model_choice = input("Ваш выбор: ").strip()
            if model_choice == "0":
                continue
            
            pendulum = DoublePendulum()
            t = np.linspace(0, 15, 1500)
            initial_state = np.random.uniform(
                [-np.pi, -1, -np.pi, -1],
                [np.pi, 1, np.pi, 1]
            )
            true_trajectory = pendulum.generate_trajectory(initial_state, t)
            
            if model_choice == "1":
                fc_prediction = predict_trajectory(fc, initial_state, t)
                plot_combined_phase_portrait(true_trajectory, fc_prediction, "FC")
            elif model_choice == "2":
                lnn_prediction = predict_trajectory(lnn, initial_state, t)
                plot_combined_phase_portrait(true_trajectory, lnn_prediction, "LNN")
            else:
                print("Неверный выбор.")
        elif choice == "6":
            if fc is None or lnn is None:
                print("Сначала загрузите или обучите модели (пункт 2)!")
                continue
            plt.plot(t, true_trajectory[:, 0], 'b-', alpha=0.5, label="Истинный q1")
            plt.plot(t, lnn_prediction[:, 0], 'r--', label="Предсказанный q1")
            plt.plot(t, true_trajectory[:, 2], 'b-', alpha=0.5, label="Истинный q2")
            plt.plot(t, lnn_prediction[:, 2], 'r--', label="Предсказанный q2")
            plt.xlabel("Время t")
            plt.ylabel("Угол (рад)")
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.plot(t, true_trajectory[:, 1], 'b-', alpha=0.5, label="Истинная скорость q1")
            plt.plot(t, lnn_prediction[:, 1], 'r--', label="Предсказанная скорость q1")
            plt.plot(t, true_trajectory[:, 3], 'b-', alpha=0.5, label="Истинная скорость q2")
            plt.plot(t, lnn_prediction[:, 3], 'r--', label="Предсказанная скорость q2")
            plt.xlabel("Время t")
            plt.ylabel("Скорость (рад/с)")
            plt.legend()
            plt.grid(True)
            plt.show()


        
        elif choice == "7":
            print("Выход из программы...")
            break
        
        else:
            print("Неверный выбор, попробуйте снова.")

if __name__ == "__main__":
    main()
