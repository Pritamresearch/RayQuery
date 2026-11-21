"""
Dataset Loader for OPV2V Real Multi-Camera Sequences.
Directly parses multi-camera calibration, relative ego poses, and 3D bounding boxes.
"""
import glob
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import yaml

NCAM = 4

def _x_to_world(cords):
    """Convert CARLA coordinate pose [x, y, z, roll, yaw, pitch] (in degrees) to 4x4 matrix."""
    x, y, z, roll, yaw, pitch = cords
    c_y = math.cos(math.radians(yaw))
    s_y = math.sin(math.radians(yaw))
    c_r = math.cos(math.radians(roll))
    s_r = math.sin(math.radians(roll))
    c_p = math.cos(math.radians(pitch))
    s_p = math.sin(math.radians(pitch))

    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r

    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r

    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


class OPV2VQueryDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        split_dir = os.path.join(cfg.data_root, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.scenario_dirs = sorted(glob.glob(os.path.join(split_dir, "*")))
        self.samples = []
        for s_dir in self.scenario_dirs:
            if not os.path.isdir(s_dir):
                continue
            # Look for vehicle directories (ego is typically first or lowest ID)
            agents = sorted([d for d in os.listdir(s_dir) if os.path.isdir(os.path.join(s_dir, d))])
            if not agents:
                continue
            ego_agent = agents[0]
            ego_dir = os.path.join(s_dir, ego_agent)
            timestamps = sorted([
                f.replace(".yaml", "") for f in os.listdir(ego_dir) if f.endswith(".yaml")
            ])
            for ts in timestamps:
                self.samples.append((s_dir, agents, ts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s_dir, agents, ts = self.samples[idx]
        
        # Limit to max_collaborators + 1 (ego)
        selected_agents = agents[: self.cfg.max_collaborators + 1]
        n_agents = len(selected_agents)
        
        images_list = []
        intrinsics_list = []
        extrinsics_list = []
        rel_poses = []
        
        # Load ego pose for reference
        ego_yaml_path = os.path.join(s_dir, selected_agents[0], f"{ts}.yaml")
        with open(ego_yaml_path, "r") as f:
            ego_data = yaml.safe_load(f)
        T_world_ego = _x_to_world(ego_data["true_ego_pos"])
        T_ego_world = np.linalg.inv(T_world_ego)

        # Parse agents
        for agent in selected_agents:
            agent_yaml_path = os.path.join(s_dir, agent, f"{ts}.yaml")
            with open(agent_yaml_path, "r") as f:
                agent_data = yaml.safe_load(f)
                
            T_world_agent = _x_to_world(agent_data["true_ego_pos"])
            T_ego_agent = T_ego_world @ T_world_agent
            dx = T_ego_agent[0, 3]
            dy = T_ego_agent[1, 3]
            dyaw = math.atan2(T_ego_agent[1, 0], T_ego_agent[0, 0])
            rel_poses.append([dx, dy, dyaw])
            
            # Load 4 camera images & parameters
            agent_imgs = []
            agent_intrins = []
            agent_extrins = []
            for c in range(NCAM):
                img_path = os.path.join(s_dir, agent, f"{ts}_camera{c}.png")
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    orig_w, orig_h = img.size
                    img = img.resize((self.cfg.img_w, self.cfg.img_h), Image.BILINEAR)
                    img_np = np.array(img, dtype=np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img_np = (img_np - mean) / std
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # (3, H, W)
                else:
                    img_tensor = torch.zeros(3, self.cfg.img_h, self.cfg.img_w, dtype=torch.float32)
                    orig_w, orig_h = self.cfg.img_w, self.cfg.img_h
                
                # Intrinsics
                if f"camera{c}" in agent_data:
                    k_mat = np.array(agent_data[f"camera{c}"]["intrinsic"], dtype=np.float32)
                    # Scale intrinsics to resized image
                    k_mat[0, :] *= (self.cfg.img_w / orig_w)
                    k_mat[1, :] *= (self.cfg.img_h / orig_h)
                    ext_mat = np.array(agent_data[f"camera{c}"]["extrinsic"], dtype=np.float32)
                else:
                    k_mat = np.identity(3, dtype=np.float32)
                    ext_mat = np.identity(4, dtype=np.float32)
                    
                agent_imgs.append(img_tensor)
                agent_intrins.append(torch.from_numpy(k_mat).float())
                agent_extrins.append(torch.from_numpy(ext_mat).float())
                
            images_list.append(torch.stack(agent_imgs))
            intrinsics_list.append(torch.stack(agent_intrins))
            extrinsics_list.append(torch.stack(agent_extrins))
            
        # Parse 3D Ground Truth Bounding Boxes from ego's perspective
        gt_boxes = []
        if "vehicles" in ego_data:
            for v_id, v_info in ego_data["vehicles"].items():
                loc = v_info["location"]
                angle = v_info["angle"]
                extent = v_info["extent"]
                # Convert from CARLA coordinates to ego vehicle frame
                v_world = _x_to_world([loc[0], loc[1], loc[2], angle[0], angle[1], angle[2]])
                v_ego = T_ego_world @ v_world
                
                cx, cy, cz = float(v_ego[0, 3]), float(v_ego[1, 3]), float(v_ego[2, 3])
                l, w, h = float(extent[0] * 2), float(extent[1] * 2), float(extent[2] * 2)
                yaw = float(math.atan2(v_ego[1, 0], v_ego[0, 0]))
                
                # Keep objects within evaluation range
                if (self.cfg.x_min <= cx <= self.cfg.x_max and
                    self.cfg.y_min <= cy <= self.cfg.y_max):
                    gt_boxes.append([cx, cy, cz, l, w, h, yaw])
                    
        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 7), dtype=torch.float32)

        return {
            "n_agents": n_agents,
            "images": torch.stack(images_list).float(),          # (N, Ncam, 3, H, W)
            "intrinsics": torch.stack(intrinsics_list).float(),  # (N, Ncam, 3, 3)
            "extrinsics": torch.stack(extrinsics_list).float(),  # (N, Ncam, 4, 4)
            "rel_poses": torch.tensor(rel_poses, dtype=torch.float32), # (N, 3)
            "gt_boxes": gt_boxes_t,                      # (G, 7)
        }
