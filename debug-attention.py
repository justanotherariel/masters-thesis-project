import torch
from pathlib import Path
import pickle
from minigrid.core.constants import DIR_TO_VEC

directory = Path("data/model-debug/")

def load_data(name: str) -> dict[str, list[torch.Tensor]]:
    return {
        "train": [
            torch.load(directory / f"{name}_0_0.pt", weights_only=True),
            torch.load(directory / f"{name}_0_1.pt", weights_only=True),
            torch.load(directory / f"{name}_0_2.pt", weights_only=True),
        ],
        "validation": [
            torch.cat([
                torch.load(directory / f"{name}_1_0.pt", weights_only=True),
                torch.load(directory / f"{name}_2_0.pt", weights_only=True),
            ]),
            torch.cat([
                torch.load(directory / f"{name}_1_1.pt", weights_only=True),
                torch.load(directory / f"{name}_2_1.pt", weights_only=True),
            ]),
            torch.cat([
                torch.load(directory / f"{name}_1_2.pt", weights_only=True),
                torch.load(directory / f"{name}_2_2.pt", weights_only=True),
            ]),
        ],
    }

attention_weights = load_data("attention")
eta = load_data("eta")
data: dict[str, list[torch.Tensor]] = {
    "train": pickle.load(open(directory / "data_TRAIN.pkl", "rb")),
    "validation": pickle.load(open(directory / "data_VALIDATION.pkl", "rb")),
}

dataset = 'train'
feature_obs, feature_action = data[dataset][0]
target_obs, target_reward = data[dataset][1]
pred_obs, pred_reward, pred_eta = data[dataset][2]
aw = attention_weights[dataset]
e = eta[dataset]

n_samples = feature_obs.shape[0]
n_heads = aw[0].shape[1]
feature_agent_pos = (feature_obs[..., 3] != 0).nonzero()


# Test
index = 845
pos = feature_agent_pos[index][1:]
print(f"Position: {pos}")
pos_front = (pos + torch.Tensor(DIR_TO_VEC[feature_obs[*feature_agent_pos[index]][3]-1])).long()

for layer_idx in range(len(attention_weights[dataset])):
    for head_idx in range(len(attention_weights[dataset][layer_idx][0])):
        print(f"Attention weights {layer_idx} (head {head_idx}): {attention_weights[dataset][layer_idx][index, head_idx, pos[0], pos[1]]}")
        print(f"Attention weights {layer_idx} (head {head_idx}) (front): {attention_weights[dataset][layer_idx][index, head_idx, pos_front[0], pos_front[1]]}")

    print(f"Eta {i}: {eta[dataset][i][index]}")
    print(f"Eta {i} (front): {eta[dataset][i][index, pos_front[0], pos_front[1]]}")
    print()

print('test')