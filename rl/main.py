import sys
import matplotlib.pyplot as plt
import os

sys.path.append("./")

from rl import rl_agent

if __name__ == "__main__":
    BOARD_SIZE = 40
    NB_TRAINING_STEPS = 10000
    NB_EVALUATION_EPISODES = 1000
    EVALUATION_DB = None # os.path.join("rl", "validation_set_upto15.pkl")
    SAVE_DIR = "model_size_40"
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPSILON_DECAY_RATE = 2000
    EPSILON_START = 0.95
    EPSILON_END = 0.05
    TARGET_MODEL_UPDATE = 100
    VALIDATION_EPISODES = 1000
    VALIDATION_INTERVAL = 5000

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    agent = rl_agent.Agent(
        board_size=BOARD_SIZE,
        epsilon_decay=EPSILON_DECAY_RATE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        evaluation_db=EVALUATION_DB,
    )
    iterations, rewards, losses = agent.train(
        NB_TRAINING_STEPS,
        batch_size=BATCH_SIZE,
        target_model_update=TARGET_MODEL_UPDATE,
        evaluation_interval=VALIDATION_INTERVAL,
        evaluation_episodes=VALIDATION_EPISODES,
    )

    average_rvr = 0
    average_avr = 0
    average_ava = 0
    average_opt = 0
    for _ in range(5):
        (
            random_v_random_winrate,
            agent_v_random_winrate,
            agent_v_agent_winrate,
            optimal_play_winrate,
        ) = agent.evaluate(NB_EVALUATION_EPISODES)
        average_rvr += random_v_random_winrate
        average_avr += agent_v_random_winrate
        average_ava += agent_v_agent_winrate
        average_opt += optimal_play_winrate

    average_rvr /= 5
    average_avr /= 5
    average_ava /= 5
    average_opt /= 5

    print(
        f"[Random vs Random] {average_rvr}\n[Agent vs Random] {average_avr}\n[Agent vs Agent] {average_ava}\n[Optimal Play] {average_opt}"
    )

    agent.save_for_deployment(os.path.join(SAVE_DIR, "model.pt"))

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(rewards, label="Average Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "average_reward.jpg"))

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(losses, label="Average Loss per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "average_losses.jpg"))
