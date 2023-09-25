import subprocess
import json
import os
import shutil

from pyparsing import List
import events as e

from pathlib import Path
from functools import cmp_to_key

from settings import MAX_AGENTS

MODELS_PARENT_DIR = './models'
MODELS_DIR = MODELS_PARENT_DIR + '/models'

DEFAULT_CONFIG = {
}

class OwnAgent:
    def __init__(self, agent_name, config, filepath):
        self.agent_name = agent_name
        self.config = config
        self.filepath = filepath

def play(scenario, rounds, own_agents, train, fill_agent = None, save_stats = True, parallel_exec = True):
    agents_str = "--agents"
    configs_str = "--agent-configs"
    for own_agent in own_agents:
        config = { 
            **own_agent.config,
            'model_filename': own_agent.filepath
        }
        filepath_no_ext = os.path.splitext(own_agent.filepath)[0]
        config_path = Path(filepath_no_ext + "_config.json")
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
            stats_str += f" \"{save_stats}\""

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
            model_path = Path(own_agent.filepath)
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

def stats_comparator(stats_0, stats_1):
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
    
def model_stats_comparator(model_stats_0, model_stats_1):
    return stats_comparator(model_stats_0[1], model_stats_1[1])

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
    stats_filenames = [str(Path(MODELS_DIR, f"{subdir}/stats{i}.json").resolve()) 
                       for i in range(len(own_agents_list))]
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

def round1(subdir = '0'):
    print("Round 1")
    own_agents_list = [
        [OwnAgent(agent_name="linear_agent", config=DEFAULT_CONFIG, 
                  filepath=str(Path(MODELS_DIR, f"{subdir}/model{i}.pt").resolve()))]
        for i in range(5)
    ]
    print("Training")
    train_models(own_agents_list, subdir=subdir, rounds=500, scenario="loot-crate", fill_agent=None, 
                 parallel_exec=True)
    print("Testing")
    stats = test_models(own_agents_list, subdir=subdir, rounds=100, scenario="loot-crate", fill_agent=None, 
                        parallel_exec=True)
    ranked_models = get_ranked_models(stats)
    return ranked_models

def round2(ranked_models, subdir = '1'):
    print("\n\nRound 2")
    init_next_round(ranked_models, new_subdir=subdir, num_best=1, repopulate_to=5, filename_template="model{}")
    own_agents_list = [
        [OwnAgent(agent_name="linear_agent", config={ 
            **DEFAULT_CONFIG,
            'override_model': False},
            filepath=str(Path(MODELS_DIR, f"{subdir}/model{i}.pt").resolve()))]
        for i in range(5)
    ]
    print("Training")
    train_models(own_agents_list, subdir=subdir, rounds=200, scenario="classic", 
                 fill_agent="rule_based_agent", parallel_exec=True)
    print("Testing")
    stats = test_models(own_agents_list, subdir=subdir, rounds=100, scenario="classic", 
                        fill_agent="rule_based_agent", parallel_exec=True)
    ranked_models = get_ranked_models(stats)
    return ranked_models

def main():
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    ranked_models = round1()
    ranked_models = round2(ranked_models)
    shutil.copy(ranked_models[0], Path(MODELS_DIR, f"best_model.pt"))

def final_ranking():
    best_model_path_objects = [Path(model_path, "best_model.pt") 
                               for model_path in Path(MODELS_PARENT_DIR).iterdir()]
    best_models = [str(best_model_path.resolve()) 
                   for best_model_path in best_model_path_objects if best_model_path.exists()]
    models_in_tournament = best_models

    def construct_own_agent(path):
        return OwnAgent(
            agent_name="linear_agent", 
            config={ 
                **DEFAULT_CONFIG,
                'override_model': False
            },
            filepath=path
        )

    round_number = 0
    while len(models_in_tournament) > 1:
        print(f"\n\nRound {round_number + 1}, {len(models_in_tournament)} agents remaining")

        Path(MODELS_PARENT_DIR, f"ft/{round_number}").mkdir(parents=True, exist_ok=True)
        group_assignments = [
            [2 * i, 2 * i + 1] for i in range(len(models_in_tournament) // 2)
        ]
        if len(models_in_tournament) % 2 == 1:
            group_assignments[-1].append(len(models_in_tournament) - 1)
        groups = [[construct_own_agent(models_in_tournament[k]) for k in group] for group in group_assignments]
        model_stats = test_models(own_agents_list=groups, subdir=f"ft/{round_number}", rounds=100,
                                  scenario="classic", fill_agent="rule_based_agent", parallel_exec=True)
        results = {}
        for (model_path, stats) in model_stats:
            model_index = models_in_tournament.index(str(model_path.resolve()))
            results[model_index] = stats
        
        round_results = []
        for group_index, group in enumerate(group_assignments):
            ranking = sorted(
                group, reverse=True, 
                key=cmp_to_key(
                    lambda o0, o1: stats_comparator(results[o0], results[o1])
                )
            )
            round_results.append([(models_in_tournament[rank], results[rank]) for rank in ranking])

        with Path(MODELS_PARENT_DIR, f"ft/{round_number}", "results.json").open('w') as round_results_file:
            json.dump(round_results, round_results_file)
        
        next_round_path = Path(MODELS_PARENT_DIR, f"ft/{round_number + 1}")
        next_models_in_tournament = []
        next_round_path.mkdir(parents=True, exist_ok=True)
        for group_index, group in enumerate(group_assignments):
            next_model_path = str(Path(next_round_path, f"model{group_index}.pt").resolve())
            shutil.copy(round_results[group_index][0][0], next_model_path)
            next_models_in_tournament.append(next_model_path)

        models_in_tournament = next_models_in_tournament
        round_number += 1

if __name__ == '__main__':
    final_ranking()