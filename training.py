import subprocess
import json
import os
import shutil

from pyparsing import List
import events as e

from pathlib import Path
from functools import cmp_to_key

from settings import MAX_AGENTS

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
}

class OwnAgent:
    def __init__(self, agent_name, config, filename):
        self.agent_name = agent_name
        self.config = config
        self.filename = filename

def play(scenario, rounds, own_agents, train, fill_agent = None, save_stats = True, parallel_exec = True):
    agents_str = "--agents"
    configs_str = "--agent-configs"
    for own_agent in own_agents:
        config = { 
            **own_agent.config,
            'model_filename': str(Path(MODELS_DIR, own_agent.filename + ".pt").resolve())
        }
        config_path = Path(MODELS_DIR, own_agent.filename + "_config.json")
        with config_path.open('w') as config_file:
            json.dump(config, config_file)
        agents_str += f" {own_agent.agent_name}"
        configs_str += f" \"{config_path.resolve()}\""
    
    if fill_agent is not None:
        agents_str += f" {fill_agent}" * (MAX_AGENTS - len(own_agents))

    train_str = f"--train {len(own_agents)}" if train else ""
    stats_str = ""
    if save_stats is not False:
        stats_str += "--save-stats"
        if save_stats is not True:
            stats_str += f" \"{Path(MODELS_DIR, save_stats).resolve()}\""

    process = subprocess.Popen(
        f"python main.py play --no-gui {agents_str} {train_str} --scenario {scenario} " +
        f"--n-rounds {rounds} {configs_str} {stats_str}"
    )

    if parallel_exec:
        return process
    
    process.wait()


def play_multiple(agents, config, scenario, rounds, save_stats = True, parallel_exec = True, filename_template = "model_{}"):
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


def load_stats(own_agents_list, stats_filenames):
    model_stats = []

    for own_agents, stats_filename in zip(own_agents_list, stats_filenames):
        stats_path = Path(MODELS_DIR, stats_filename)
        with stats_path.open('r') as stats_file:
            stats = json.load(stats_file)
        stats_by_agent = stats['by_agent']

        for i, own_agent in enumerate(own_agents):
            model_path = Path(MODELS_DIR, own_agent.filename + ".pt")
            key = own_agent.agent_name
            if key not in stats_by_agent:
                for i in range(MAX_AGENTS):
                    numbered_key = f"{key}_{i}"
                    if numbered_key in stats_by_agent:
                        key = numbered_key
                        break

                if key not in stats_by_agent:
                    raise ValueError(f"{own_agent.agent_name} not in stats['by_agent']")

            model_stats.append((model_path, stats_by_agent[key]))
            del stats_by_agent[key]

    return model_stats

    
def model_stats_comparator(model_stats_0, model_stats_1):
    stats_0, stats_1 = model_stats_0[1], model_stats_1[1]

    if stats_0.get('score', 0) < stats_1.get('score', 0):
        return -1
    if stats_0.get('score', 0) > stats_1.get('score', 0):
        return +1
    
    if stats_0.get('suicides', 0) < stats_1.get('suicides', 0):
        return +1
    if stats_0.get('suicides', 0) > stats_1.get('suicides', 0):
        return -1
    
    if stats_0.get('invalid', 0) < stats_1.get('invalid', 0):
        return +1
    if stats_0.get('invalid', 0) > stats_1.get('invalid', 0):
        return -1
    
    return 0


def get_ranked_models(model_stats):
    sorted_model_stats = sorted(model_stats, reverse=True, key=cmp_to_key(model_stats_comparator))
    return list(map(lambda ms: ms[0], sorted_model_stats))


def copy_best(model, filename_template = "model_{}"):
    shutil.copy(model, Path(MODELS_DIR, filename_template.format("best") + ".pt"))


def train_models(own_agents_list, subdir, rounds, scenario, fill_agent = None, parallel_exec = False):
    Path(MODELS_DIR, subdir).mkdir(parents=True, exist_ok=True)
    processes = []
    for own_agents in own_agents_list:
        processes.append(play(
            scenario=scenario,
            rounds=rounds, 
            own_agents=own_agents,
            train=True,
            fill_agent=fill_agent,
            save_stats=False,
            parallel_exec=parallel_exec
        ))

    if parallel_exec:
        for process in processes:
            process.wait()

def test_models(own_agents_list, subdir, rounds, scenario, fill_agent = None, parallel_exec = False):
    stats_filenames = [f"{subdir}/stats{i}.json" for i in range(len(own_agents_list))]
    processes = []
    for own_agents, stats_filename in zip(own_agents_list, stats_filenames):
        processes.append(play(
            scenario=scenario,
            rounds=rounds, 
            own_agents=own_agents,
            train=False,
            fill_agent=fill_agent,
            save_stats=stats_filename,
            parallel_exec=parallel_exec
        ))

    if parallel_exec:
        for process in processes:
            process.wait()
    
    stats = load_stats(own_agents_list, stats_filenames)
    return stats


def init_next_round(ranked_models, new_subdir, num_best: int, repopulate_to: int, filename_template = "model{}"):      
    dst_dir = Path(MODELS_DIR, new_subdir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i in range(repopulate_to):
        src = ranked_models[i % num_best]
        dst = Path(dst_dir, f"{filename_template.format(i)}.pt")
        shutil.copy(src, dst)

def main():
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    config = DEFAULT_CONFIG
    num_agents = 3
    num_best = 1
    num_rounds = 1

    print("Round 1")
    own_agents_list = [
        [OwnAgent(agent_name="linear_agent", config=DEFAULT_CONFIG, filename=f"0/model{i}"),]
        for i in range(8)
    ]
    print("Training")
    train_models(own_agents_list, subdir='0', rounds=5, scenario="loot-crate", fill_agent=None, 
                 parallel_exec=False)
    print("Testing")
    stats = test_models(own_agents_list, subdir='0', rounds=1, scenario="loot-crate", fill_agent=None, 
                        parallel_exec=False)
    ranked_models = get_ranked_models(stats)



    print("\n\nRound 2")
    init_next_round(ranked_models, new_subdir='1', num_best=1, repopulate_to=8, filename_template="model{}")
    own_agents_list = [
        [OwnAgent(agent_name="linear_agent", config={ 
            **DEFAULT_CONFIG, 
            'model_filename': str(Path(MODELS_DIR, f"1/model{i}.pt").resolve()), 
            'override_model': False},
            filename=f"0/model{i}")]
        for i in range(8)
    ]
    print("Training")
    train_models(own_agents_list, subdir='1', rounds=2, scenario="classic", 
                 fill_agent="rule_based_agent", parallel_exec=False)
    print("Testing")
    stats = test_models(own_agents_list, subdir='1', rounds=1, scenario="classic", 
                        fill_agent="rule_based_agent", parallel_exec=False)
    ranked_models = get_ranked_models(stats)
    shutil.copy(ranked_models[0], Path(MODELS_DIR, f"best_model.pt"))



    """
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
    """

if __name__ == '__main__':
    main()