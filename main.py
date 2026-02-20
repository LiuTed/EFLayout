import json
import yaml
import argparse
import tqdm

import numpy as np
import torch
from torch.distributions import Categorical

from configuration import EFLayoutConfig
from modeling import EFLayoutModelBackbone, EFLayoutActorCritic
from game import generate_new_game

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    
    args = parser.parse_args()
    return args

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    else:
        return obj

def main():
    args = parse_args()
    with open(args.train, "r") as f:
        train_config = yaml.safe_load(f)
    device = torch.device(train_config.get("device", "cpu"))

    with open(args.model, "r") as f:
        model_config = json.load(f)
        model_config = EFLayoutConfig(**model_config)
    
        model_backbone = EFLayoutModelBackbone(model_config).to(device)
        model_ac = EFLayoutActorCritic(model_config).to(device)
    
    print(model_backbone)
    print(model_ac)
    
    optimizer = torch.optim.AdamW(
        list(model_backbone.parameters()) + list(model_ac.parameters()),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"])
    )
    
    for iteration in tqdm.trange(1, train_config["num_iterations"] + 1, desc='Iterations', leave=True, unit='iter'):
        states = []
        rewards = []
        dones = []
        values = []
        actions = []
        logprobs = []
        masks = []
        
        final_state = None
        bar = tqdm.tqdm(desc='Collecting samples', leave=False, total=train_config["buffer_size"], unit='samples')
        with torch.no_grad():
            while len(states) < train_config["buffer_size"]:
                env = generate_new_game(0)
                done = False
                while not done and len(states) < train_config["buffer_size"]:
                    state = env.get_states()
                    state = move_to_device(state, device)
                    
                    bb = model_backbone(**state)
                    critic = model_ac.get_critic(**bb)
                    
                    action_mask = env.get_action_mask()
                    action_logits, _ = model_ac.get_action(**bb, action_mask=action_mask)
                    action_cate = Categorical(logits=action_logits)
                    action = action_cate.sample()
                    
                    if action == 0:
                        machine_mask = env.get_machine_mask()
                        machine_logits, _ = model_ac.get_machine(**bb, machine_mask=machine_mask)
                        machine_cate = Categorical(logits=machine_logits)
                        machine = machine_cate.sample()
                        
                        spatial_machine_mask = env.get_spatial_machine_mask(machine.item())
                        lo_logits, _ = model_ac.get_location_orientation(
                            **bb,
                            selected=machine.unsqueeze(0),
                            spatial_machine_mask=spatial_machine_mask
                        )
                        lo_cate = Categorical(logits=lo_logits)
                        lo = lo_cate.sample()

                        reward = env.put_machine_model(machine.item(), lo.item())
                        
                        action_tmp = ((action,), (action, machine), (action, machine, lo))
                        logits_tmp = (action_cate.log_prob(action), machine_cate.log_prob(machine), lo_cate.log_prob(lo))
                        mask_tmp = (action_mask, machine_mask, spatial_machine_mask)
                    elif action == 1:
                        belt_mask = env.get_spatial_belt_mask()
                        belt_logits, _ = model_ac.get_belt_location_orientation(**bb, spatial_belt_mask=belt_mask)
                        belt_cate = Categorical(logits=belt_logits)
                        belt = belt_cate.sample()
                        
                        reward = env.put_belt_model(belt.item())

                        action_tmp = ((action,), (action, belt))
                        logits_tmp = (action_cate.log_prob(action), belt_cate.log_prob(belt))
                        mask_tmp = (action_mask, belt_mask)
                    elif action == 2:
                        power_mask = env.get_spatial_power_mask()
                        power_logits, _ = model_ac.get_power_location(**bb, spatial_power_mask=power_mask)
                        power_cate = Categorical(logits=power_logits)
                        power = power_cate.sample()
                        
                        reward = env.put_power_model(power.item())
                        
                        action_tmp = ((action,), (action, power))
                        logits_tmp = (action_cate.log_prob(action), power_cate.log_prob(power))
                        mask_tmp = (action_mask, power_mask)
                    else:
                        raise RuntimeError(f"Unknown action {action}")
                    
                    done = env.finished
                    
                    actions.extend(action_tmp)
                    # rewards.extend([0.] * (len(action_tmp) - 1) + [reward])
                    rewards.extend([reward / len(action_tmp)] * len(action_tmp))
                    values.extend([critic] * len(action_tmp))
                    states.extend([state] * len(action_tmp))
                    dones.extend([False] * (len(action_tmp) - 1) + [done])
                    logprobs.extend(logits_tmp)
                    masks.extend(mask_tmp)

                    bar.update(len(action_tmp))
                final_state = move_to_device(env.get_states(), device)
        values = torch.tensor(values, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        
        def collate(batch):
            if isinstance(batch[0], dict):
                basic = {
                    k: collate([item[k] for item in batch]) for k in batch[0] if k != "edge_index"
                }
                if "edge_index" in batch[0]:
                    basic["edge_index"] = torch.cat([item["edge_index"] for item in batch], dim=1)
                return basic
            elif isinstance(batch[0], torch.Tensor):
                return torch.cat(batch, dim=0)
            else:
                return torch.utils.data.default_collate(batch)
        with torch.no_grad():
            bb = model_backbone(**final_state)
            critics = model_ac.get_critic(**bb)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = critics
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + train_config["gamma"] * nextnonterminal * nextvalues - values[t]
                advantages[t] = lastgaelam = delta + nextnonterminal * lastgaelam * train_config["gamma"] * train_config["gae_lambda"]
            returns = advantages + values

        for epoch in tqdm.trange(train_config["update_epochs"], desc='Updating', leave=False, unit='epochs'):
            indices = np.arange(len(rewards))
            np.random.shuffle(indices)

            bar = tqdm.trange(
                0, len(rewards), train_config["batch_size"],
                desc='Batches',
                leave=False,
                unit='batches'
            )
            for start in bar:
                end = start + train_config["batch_size"]
                batch_indices = indices[start:end]
                
                optimizer.zero_grad()
                
                subaction_indices = {
                    "action": {
                        "buffer": [],
                        "judge": lambda x: len(actions[x]) == 1,
                        "actor": model_ac.get_action,
                        "extra_kwargs": lambda mask, acts: {"action_mask": mask},
                    },
                    "machine": {
                        "buffer": [],
                        "judge": lambda x: actions[x][0] == 0 and len(actions[x]) == 2,
                        "actor": model_ac.get_machine,
                        "extra_kwargs": lambda mask, acts: {"machine_mask": mask},
                    },
                    "machine_lo": {
                        "buffer": [],
                        "judge": lambda x: actions[x][0] == 0 and len(actions[x]) == 3,
                        "actor": model_ac.get_location_orientation,
                        "extra_kwargs": lambda mask, acts: {"spatial_machine_mask": mask, "selected": torch.tensor([a[1] for a in acts], device=device)},
                    },
                    "belt": {
                        "buffer": [],
                        "judge": lambda x: actions[x][0] == 1 and len(actions[x]) == 2,
                        "actor": model_ac.get_belt_location_orientation,
                        "extra_kwargs": lambda mask, acts: {"spatial_belt_mask": mask},
                    },
                    "power": {
                        "buffer": [],
                        "judge": lambda x: actions[x][0] == 2 and len(actions[x]) == 2,
                        "actor": model_ac.get_power_location,
                        "extra_kwargs": lambda mask, acts: {"spatial_power_mask": mask},
                    },
                }
                for idx in batch_indices:
                    for k, v in subaction_indices.items():
                        if v["judge"](idx):
                            v["buffer"].append(idx)
                
                sumloss = 0.
                for k, v in subaction_indices.items():
                    if len(v["buffer"]) == 0:
                        continue
                    bstate = collate([states[idx] for idx in v["buffer"]])
                    tmp = [masks[idx].shape for idx in v["buffer"]]
                    bmask = collate([masks[idx] for idx in v["buffer"]])
                    bb = model_backbone(**bstate)
                    kwargs = v["extra_kwargs"](bmask, [actions[idx] for idx in v["buffer"]])
                    _, logits_list = v["actor"](**bb, **kwargs)
                    
                    new_ratios = []
                    entropy = 0.
                    for idx, logits in zip(v["buffer"], logits_list):
                        cate = Categorical(logits=logits)
                        logratio = cate.log_prob(actions[idx][-1]) - logprobs[idx]
                        new_ratios.append(logratio.exp())
                        entropy += cate.entropy()
                    new_ratios = torch.stack(new_ratios, dim=0)
                    advs = advantages[v["buffer"]]
                    
                    if train_config['normalize_advantages']:
                        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                    
                    pg_loss1 = -new_ratios * advs
                    pg_loss2 = -new_ratios.clamp(1.-train_config['clip_range'], 1.+train_config['clip_range']) * advs
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    new_critics = model_ac.get_critic(**bb)
                    v_loss = 0.5 * ((new_critics - returns[v["buffer"]]) ** 2).mean()

                    loss = pg_loss + v_loss * train_config['value_coef'] - train_config['entropy_coef'] * entropy.mean()
                    sumloss += loss.item() * len(v["buffer"])
                    loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model_ac.parameters(), train_config['max_grad_norm'])
                torch.nn.utils.clip_grad_norm_(model_backbone.parameters(), train_config['max_grad_norm'])
                
                avgloss = sumloss / len(indices)
                bar.set_postfix({'loss': avgloss}, refresh=False)
                
                optimizer.step()

        if epoch % train_config['save_epochs'] == 0:
            torch.save(model_ac.state_dict(), f'{train_config["save_dir"]}/model_ac_{epoch}.pt')
            torch.save(model_backbone.state_dict(), f'{train_config["save_dir"]}/model_backbone_{epoch}.pt')
            
if __name__ == '__main__':
    main()
