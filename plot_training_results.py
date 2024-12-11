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
        Genera todos los gráficos juntos en una sola figura con subplots y muestra los valores en puntos específicos.
        """
        df = pd.DataFrame(self.metrics)

        # Crear una figura con 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Training Metrics", fontsize=16)

        # Función para agregar anotaciones en los puntos
        def annotate_points(ax, x, y):
            for i in range(len(x)):
                if i % 10 == 0:  # Mostrar valores cada 10 episodios
                    ax.annotate(f"{y[i]:.2f}", (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha="center")

        # Gráfico 1: Recompensa total por episodio
        axes[0, 0].plot(df["episode"], df["total_reward"], label="Total Reward", color="blue")
        annotate_points(axes[0, 0], df["episode"], df["total_reward"])
        axes[0, 0].set_title("Reward Per Episode")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].legend()

        # Gráfico 2: Pérdida promedio
        axes[0, 1].plot(df["episode"], df["loss"], label="Loss", color="red")
        annotate_points(axes[0, 1], df["episode"], df["loss"])
        axes[0, 1].set_title("Loss Per Episode")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()

        # Gráfico 3: Gradientes promedio
        axes[1, 0].plot(df["episode"], df["mean_gradients"], label="Mean Gradients", color="green")
        annotate_points(axes[1, 0], df["episode"], df["mean_gradients"])
        axes[1, 0].set_title("Mean Gradients Per Episode")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Gradients")
        axes[1, 0].legend()

        # Gráfico 4: Entropía promedio
        axes[1, 1].plot(df["episode"], df["entropy"], label="Entropy", color="purple")
        annotate_points(axes[1, 1], df["episode"], df["entropy"])
        axes[1, 1].set_title("Entropy Per Episode")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Entropy")
        axes[1, 1].legend()

        # Ajustar espaciado entre subplots
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Mostrar figura
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
