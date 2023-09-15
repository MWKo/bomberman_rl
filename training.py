import subprocess
import json
import os
import shutil

from pyparsing import List
import events as e

from pathlib import Path
from functools import cmp_to_key

MODELS_DIR = './models'

DEFAULT_CONFIG = {
    'override_model': True,
    'exploration': {
        'epsilon': 0.1,
        'action_probabilities': [.2, .2, .2, .2, .1, .1]
    },
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

    model_filenames = [filename_template.format(i) for i in range(len(agents))]
    model_paths = [Path(MODELS_DIR, model_filename + ".pt") for model_filename in model_filenames]
    config_paths = [Path(MODELS_DIR, model_filename + "_config.json") for model_filename in model_filenames]
    stats_paths = [Path(MODELS_DIR, model_filename + "_stats.json") for model_filename in model_filenames]
    for i, agent in enumerate(agents):
        config_model = { **config, 'model_filename': f"{model_paths[i].resolve()}" }
        with config_paths[i].open('w') as config_file:
            json.dump(config_model, config_file)

        save_stats_arg = f" --save-stats \"{stats_paths[i].resolve()}\"" if save_stats else ""

        training_processes.append(subprocess.Popen(
            f"python main.py play --no-gui --train 1 --agents {agent} --scenario {scenario} " +
            f"--n-rounds {rounds} --agent-configs \"{config_paths[i].resolve()}\"" + save_stats_arg
        ))
        if not parallel_exec:
            process.wait()

    if parallel_exec:
        for process in training_processes:
            process.wait()
    
    if not save_stats:
        return None

    agent_stats = []
    for agent, stats_path in zip(agents, stats_paths):
        with stats_path.open('r') as stats_file:
            stats = json.load(stats_file)
        agent_stats.append(stats['by_agent'][agent])
    return zip(model_paths, agent_stats)

    
def model_stats_comparator(model_stats_0, model_stats_1):
    stats_0, stats_1 = model_stats_0[1], model_stats_1[1]

    if stats_0['score'] < stats_1['score']:
        return -1
    if stats_0['score'] > stats_1['score']:
        return +1
    
    if stats_0['suicides'] < stats_1['suicides']:
        return +1
    if stats_0['suicides'] > stats_1['suicides']:
        return -1
    
    if stats_0['invalid'] < stats_1['invalid']:
        return +1
    if stats_0['invalid'] > stats_1['invalid']:
        return -1
    
    return 0


def get_ranked_models(model_stats):
    sorted_model_stats = sorted(model_stats, reverse=True, key=cmp_to_key(model_stats_comparator))
    return list(map(lambda ms: ms[0], sorted_model_stats))


def copy_best(model, filename_template = "model_{}"):
    shutil.copy(model, Path(MODELS_DIR, filename_template.format("best") + ".pt"))


def select_repopulate(models, num_best: int, repopulate_to: int, filename_template = "model_{}"):
    for model in models[num_best:]:
        model.unlink()
    
    models = models[:num_best]
    temp_modelpaths = [Path(MODELS_DIR, filename_template.format(i) + ".tmp") for i in range(len(models))]
    new_modelpaths = [Path(MODELS_DIR, filename_template.format(i) + ".pt") for i in range(len(models))]
    for model, temp_modelpath in zip(models, temp_modelpaths):
        model.rename(temp_modelpath)
    for temp_modelpath, new_modelpath in zip(temp_modelpaths, new_modelpaths):
        temp_modelpath.rename(new_modelpath)
    
    for i in range(num_best, repopulate_to):
        src = new_modelpaths[i % num_best]
        dst = Path(MODELS_DIR, filename_template.format(i) + ".pt")
        shutil.copy(src, dst)


def main():
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    config = DEFAULT_CONFIG
    num_agents = 8
    num_best = 3
    num_rounds = 5

    print("Round 1")
    model_stats = train(
        agents=["linear_agent"] * num_agents,
        config=DEFAULT_CONFIG, 
        scenario="loot-crate",
        rounds=500
    )
    models = get_ranked_models(model_stats)
    copy_best(models[0])

    for i in range(1, num_rounds):
        print(f"Round {i + 1}")
        select_repopulate(models=models, num_best=num_best, repopulate_to=num_agents)
        config = { 
            **config, 
            'override_model': False,
            'learning_rate': config['learning_rate'] / 2
        }
        model_stats = train(
            agents=["linear_agent"] * num_agents,
            config=DEFAULT_CONFIG, 
            scenario="loot-crate",
            rounds=500
        )
        models = get_ranked_models(model_stats)
        copy_best(models[0])

if __name__ == '__main__':
    main()