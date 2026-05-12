import torch
import torch.nn as nn
from monai.networks.nets import resnet


def create_model(cfg, device):
    model = resnet.resnet34(
        spatial_dims=cfg.model.spatial_dims,
        n_input_channels=cfg.model.n_channels,
        num_classes=cfg.model.num_classes,
    )

    if cfg.model.dropout_rate > 0:
        if hasattr(model, "fc"):
            model.fc = nn.Sequential(nn.Dropout(cfg.model.dropout_rate), model.fc)
        elif hasattr(model, "classifier"):
            model.classifier = nn.Sequential(nn.Dropout(cfg.model.dropout_rate), model.classifier)
        else:
            print("[Warning] Classification layer not found; dropout was not added.")

    return model.to(device)


def adapt_pretrained_weights(state_dict, n_channels: int):
    if "conv1.weight" in state_dict:
        original_weight = state_dict["conv1.weight"]
        if original_weight.shape[1] == 1 and n_channels == 2:
            state_dict["conv1.weight"] = original_weight.repeat(1, 2, 1, 1, 1) / 2.0
        elif original_weight.shape[1] != n_channels:
            print(
                f"[Warning] Pretrained input channels = {original_weight.shape[1]}, "
                f"target input channels = {n_channels}."
            )
    return state_dict


def load_pretrained_if_available(model, cfg, device):
    pretrained = cfg.paths.pretrained
    if not pretrained:
        print("No pretrained checkpoint configured.")
        return model

    import os
    if not os.path.exists(pretrained):
        print("Pretrained checkpoint not found. Training from scratch.")
        return model

    print(f"Loading pretrained weights: {pretrained}")
    state_dict = torch.load(pretrained, map_location=device)
    state_dict = adapt_pretrained_weights(state_dict, cfg.model.n_channels)

    for key in list(state_dict.keys()):
        if "fc" in key or "classifier" in key:
            state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights loaded.")
    return model
