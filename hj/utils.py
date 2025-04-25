import gymnasium as gym
import torch, numpy as np, torch.nn as nn
import tianshou as ts
from formularone import SafeGymnasium
import os
import sys
import ruamel.yaml as yaml
sys.path.append("D:/Code/WashU/vision_project/dreamerv3-torch/")
from dreamer_nom import Dreamer
import pathlib
import argparse
import tools

def load_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
        )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", "safe_gymnasium"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args()
    # train_envs = SafeGymnasium(name="SafetyCarFormulaOne1-v0")
    # train_envs = SafeGymnasium(name="SafetyCarFormulaOne1Vision-v0")
    train_envs = SafeGymnasium(name="SafetyCarGoal2Vision-v0")
    acts = train_envs.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    agent = Dreamer(
            train_envs.observation_space,
            train_envs.action_space,
            config,
            None,
            None,
        ).to(config.device)
    state_dict = torch.load("cost.pt")
    new_state_dict = {}
    for key in list(state_dict["agent_state_dict"].keys()):
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            new_state_dict[deal_key] = state_dict["agent_state_dict"][key]
    agent.load_state_dict(new_state_dict)
    agent_reach = Dreamer(
            train_envs.observation_space,
            train_envs.action_space,
            config,
            None,
            None,
        ).to(config.device)
    state_dict = torch.load("reaching.pt")
    new_state_dict = {}
    for key in list(state_dict["agent_state_dict"].keys()):
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            new_state_dict[deal_key] = state_dict["agent_state_dict"][key]
    agent_reach.load_state_dict(new_state_dict)
    return agent, agent_reach