import subprocess
import json
import os
import events as e

from pathlib import Path

MODELS_DIR = './models'

DEFAULT_CONFIG = {
    'override_model': True,
    'exploration': {
        'epsilon': 0.1,
        'action_probabilities': [.2, .2, .2, .2, .1, .1]
    },
    'exploration_probability': 0.1,
    'learning_rate': 0.01,
    'gamma': 0.98,
    'learning_stepsize': 1,
    'rewards': {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.CRATE_DESTROYED: 0.3,
        e.KILLED_SELF: -3,
    }
}

def train(agents, config, scenario, rounds, save_stats = True, parallel_exec = True, filename_template = "model_{}"):
    training_processes = []

    for i, agent in enumerate(agents):
        model_filename = filename_template.format(i)
        model_path = Path(MODELS_DIR, model_filename + ".pt")
        config_path = Path(MODELS_DIR, model_filename + "_config.json")
        stats_path = Path(MODELS_DIR, model_filename + "_stats.json")

        config_model = { **config, 'model_filename': f"{model_path.resolve()}" }
        with config_path.open('w') as config_file:
            json.dump(config_model, config_file)

        save_stats_arg = f" --save-stats \"{stats_path.resolve()}\"" if save_stats else ""

        training_processes.append(subprocess.Popen(
            f"python main.py play --no-gui --train 1 --agents {agent} --scenario {scenario} " +
            f"--n-rounds {rounds} --agent-configs \"{config_path.resolve()}\"" + save_stats_arg
        ))
        if not parallel_exec:
            process.wait()

    if parallel_exec:
        for process in training_processes:
            process.wait()
    


def main():
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    train(
        agents=["linear_agent"] * 4,
        config=DEFAULT_CONFIG, 
        scenario="loot-crate",
        rounds=100,
        save_stats=True,
        parallel_exec=True,
        filename_template="model_{}"
    )


if __name__ == '__main__':
    main()