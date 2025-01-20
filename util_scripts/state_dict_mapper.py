import argparse

import torch
import json

from src.ldm.modules.diffusionmodules.openaimodel import main_setup, load_model_from_config


def _create_mapper(state_dict1, state_dict2):
    mapper = {}
    reverse_mapper = {}
    unmatched_layers = []

    # Get the layer names and weights from both state dicts
    layers1 = list(state_dict1.items())
    layers2 = list(state_dict2.items())

    # Iterate through layers in state_dict1
    for name1, tensor1 in layers1:
        matched = False
        # Iterate through layers in state_dict2 to find the match
        for name2, tensor2 in layers2:
            if torch.equal(tensor1, tensor2):
                if name2 in reverse_mapper:
                    print(f"Error: Multiple layers in state_dict1 map to the same layer in state_dict2: {name2}")
                    return None
                mapper[name1] = name2
                reverse_mapper[name2] = name1
                matched = True
                break

        if not matched:
            unmatched_layers.append(name1)

    if unmatched_layers:
        print("Error: Unmatched layers in state_dict1:", unmatched_layers)
        return None

    if len(mapper) != len(state_dict1):
        print("Error: Not all layers in state_dict1 have been mapped.")
        return None

    return mapper

def get_component_mapper():
    EXP_PATH="./cxr_phrase_grounding/src/ldm/modules/diffusionmodules/default_cfg.py"
    config = main_setup(EXP_PATH)
    diffusers_model = torch.load('./checkpoint/diffusers.pth')
    lightning_model = load_model_from_config(config, './checkpoint/lightning.pth')
    return _create_mapper(lightning_model.state_dict(), diffusers_model.state_dict())


