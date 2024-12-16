import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compare_tensors(tensor1, tensor2, description):
    """
    Compara dos tensores para verificar si son iguales.
    """
    if not torch.equal(tensor1, tensor2):
        print(f"[ERROR] {description} NO coinciden.")
        return False
    else:
        print(f"[OK] {description} coinciden.")
    return True


def compare_optimizers(optimizer1, optimizer2):
    """
    Compara dos optimizadores verificando sus parámetros.
    """
    state1 = optimizer1.state_dict()
    state2 = optimizer2.state_dict()

    for key in state1:
        if key not in state2:
            print(f"[ERROR] Clave {key} no está en el segundo optimizador.")
            return False

        if isinstance(state1[key], torch.Tensor) and isinstance(state2[key], torch.Tensor):
            if not torch.equal(state1[key], state2[key]):
                print(f"[ERROR] Optimizer param {key} NO coinciden.")
                return False
        else:
            if state1[key] != state2[key]:
                print(f"[ERROR] Optimizer param {key} NO coinciden.")
                return False

    print("[OK] Optimizer coincide.")
    return True


def test_save_load():
    """
    Prueba para verificar el método save y load del PPOAgent.
    """
    # Importar tu clase PPOAgent aquí
    from Agents.PPOAgent import PPOAgent

    # Crear agentes para la prueba
    state_dim = 4
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim)

    # Guardar el estado actual del agente
    save_path = "test_model.pth"
    agent.save(save_path)

    # Crear otro agente para cargar
    loaded_agent = PPOAgent(state_dim, action_dim)
    loaded_agent.load(save_path)

    # Comparar los valores de epsilon
    if agent.epsilon != loaded_agent.epsilon:
        print(f"[ERROR] epsilon NO coinciden.")
    else:
        print("[OK] epsilon coinciden.")

    # Comparar los parámetros de las políticas
    print("\nComparando los parámetros de la política:")
    for param1, param2 in zip(agent.policy.parameters(), loaded_agent.policy.parameters()):
        compare_tensors(param1, param2, "policy param")

    print("\nComparando los parámetros de la política antigua:")
    for param1, param2 in zip(agent.policy_old.parameters(), loaded_agent.policy_old.parameters()):
        compare_tensors(param1, param2, "policy_old param")

    print("\nComparando el estado del optimizador:")
    compare_optimizers(agent.optimizer, loaded_agent.optimizer)

    print("\nPrueba de guardado y carga finalizada.")


if __name__ == "__main__":
    test_save_load()
