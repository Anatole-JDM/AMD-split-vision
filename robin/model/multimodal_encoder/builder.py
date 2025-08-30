import os
from .clip_encoder import CLIPVisionTower
from .open_clip import OpenCLIPVisionTower
from .timm_vision import TimmVisionTower
from .encoder_info import CLIP_COMPATIBLE_MODELS, OPENCLIP_CONFIG_MAP
from .split_vision_tower import MultiVisionTower

def build_vision_tower(vision_tower_cfg, split_shape=None, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    if "SplitVisionTransformer" in vision_tower:
        if vision_tower == "SplitVisionTransformer":
            split_shape = [2, 2]
        if split_shape == None:
            split_shape = [int(vision_tower[-3]), int(vision_tower[-1])]
        tile_count = split_shape[0]*split_shape[1]

        vision_towers=["/work1/aroger/shared/downloaded_models/ViT-SO400M-14-SigLIP-384" for x in range(0,tile_count)]

        vision_tower_list = []
        for vt in vision_towers:
            if os.path.exists(vt):
                name = vt.split("/")[-1].lower()
                name = name.replace("hf-hub:", "")
                
                if name in CLIP_COMPATIBLE_MODELS:
                    vision_tower_list.append(CLIPVisionTower(vt, args=vision_tower_cfg, **kwargs))
                elif "dino" in str(vt).lower():
                    vision_tower_list.append(TimmVisionTower(vt, args=vision_tower_cfg, **kwargs))
                elif name in OPENCLIP_CONFIG_MAP.keys():
                    vision_tower_list.append(OpenCLIPVisionTower(vt, args=vision_tower_cfg, **kwargs))
                else:
                    raise ValueError(f'Unknown local vision tower in list: {vt}')
            else:
                vt_tmp = vt.lower().replace("hf-hub:", "")
                
                if vt_tmp.startswith("openai") or vt_tmp.startswith("facebook"):
                    vision_tower_list.append(CLIPVisionTower(vt, args=vision_tower_cfg, **kwargs))
                elif "dino" in vt_tmp:
                    vision_tower_list.append(TimmVisionTower(vt, args=vision_tower_cfg, **kwargs))
                else:
                    vision_tower_list.append(OpenCLIPVisionTower(vt, args=vision_tower_cfg, **kwargs))

        return MultiVisionTower(vision_tower_list, args=vision_tower_cfg, split_shape = split_shape, **kwargs)

    else:
        if os.path.exists(vision_tower):
            name = vision_tower.split("/")[-1].lower()
            name = name.replace("hf-hub:", "")

            if name in CLIP_COMPATIBLE_MODELS:
                return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            elif "dino" in str(vision_tower).lower():
                return TimmVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            elif name in OPENCLIP_CONFIG_MAP.keys():
                return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                print(f'Local model not handled in configs (might crash): {vision_tower}')
                return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            vision_tower_tmp = vision_tower.lower()
            vision_tower_tmp = vision_tower_tmp.replace("hf-hub:", "")

            if vision_tower_tmp.startswith("openai") or vision_tower_tmp.startswith("facebook"):
                vision_tower = vision_tower.replace("hf-hub:", "")
                return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            elif "dino" in vision_tower_tmp:
                return TimmVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


