import matplotlib.pyplot as plt
import pandas as pd

class MetricsPlotter:
    def __init__(self):
        # Diccionario para guardar métricas
        self.metrics = {
            "episode": [],
            "total_reward": [],
            "mean_gradients": [],
            "loss": [],
            "entropy": []
        }

    def store_metrics(self, episode, total_reward, mean_gradients, loss, entropy):
        """
        Almacena las métricas de cada episodio en el diccionario de métricas.
        """
        self.metrics["episode"].append(episode)
        self.metrics["total_reward"].append(total_reward)
        self.metrics["mean_gradients"].append(mean_gradients)
        self.metrics["loss"].append(loss)
        self.metrics["entropy"].append(entropy)

    def plot_all(self):
        """
        Genera todos los gráficos en ventanas separadas.
        """
        df = pd.DataFrame(self.metrics)

        # Función para agregar anotaciones en los puntos
        def annotate_points(ax, x, y):
            for i in range(len(x)):
                if i % 10 == 0:  # Mostrar valores cada 10 episodios
                    ax.annotate(f"{y[i]:.2f}", (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha="center")

        # Gráfico 1: Recompensa total por episodio
        plt.figure(figsize=(8, 6))
        plt.plot(df["episode"], df["total_reward"], label="Total Reward", color="blue")
        annotate_points(plt.gca(), df["episode"], df["total_reward"])
        # Mostrar el valor máximo en el gráfico
        max_reward = max(df["total_reward"])
        max_index = df["total_reward"].idxmax()
        plt.annotate(f"Max: {max_reward:.2f}", 
                     (df["episode"][max_index], max_reward), 
                     textcoords="offset points", xytext=(0, 10), ha="center", color="red")
        plt.title("Reward Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.show()

        # Gráfico 2: Pérdida promedio
        plt.figure(figsize=(8, 6))
        plt.plot(df["episode"], df["loss"], label="Loss", color="red")
        annotate_points(plt.gca(), df["episode"], df["loss"])
        plt.title("Loss Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Gráfico 3: Gradientes promedio
        plt.figure(figsize=(8, 6))
        plt.plot(df["episode"], df["mean_gradients"], label="Mean Gradients", color="green")
        annotate_points(plt.gca(), df["episode"], df["mean_gradients"])
        plt.title("Mean Gradients Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Gradients")
        plt.legend()
        plt.show()

        # Gráfico 4: Entropía promedio
        plt.figure(figsize=(8, 6))
        plt.plot(df["episode"], df["entropy"], label="Entropy", color="purple")
        annotate_points(plt.gca(), df["episode"], df["entropy"])
        plt.title("Entropy Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Entropy")
        plt.legend()
        plt.show()

    def plot_action_probabilities(self, action_probs):
        """
        Genera un gráfico de barras de las probabilidades de acción.
        """
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(action_probs)), action_probs, color="blue")
        plt.xlabel("Actions")
        plt.ylabel("Probability")
        plt.title("Action Probabilities")
        plt.show()
