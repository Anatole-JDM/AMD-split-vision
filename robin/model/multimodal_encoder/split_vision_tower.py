import torch
import torch.nn as nn
from .clip_encoder import CLIPVisionTower
from .open_clip import OpenCLIPVisionTower
from .timm_vision import TimmVisionTower
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Removes the limit


def split_image(image, grid_size, include_full=True):
    width, height = image.size
    h_split, w_split = grid_size
    h_chunk = height // h_split
    w_chunk = width // w_split

    image_parts = []
    for i in range(h_split):
        for j in range(w_split):
            left = j * w_chunk
            upper = i * h_chunk
            right = (j + 1) * w_chunk
            lower = (i + 1) * h_chunk

            part = image.crop((left, upper, right, lower))
            image_parts.append(part)

    if include_full:
        resized_full_image = image.resize((w_chunk, h_chunk))  # Resize full image to match tile size
        image_parts.append(resized_full_image)

    return image_parts


def adaptive_pool_features(features, target_size):
    """Applies AdaptiveAvgPool1d to make all feature maps have the same spatial dimension."""
    return [F.adaptive_avg_pool1d(feat.permute(0, 2, 1), target_size).permute(0, 2, 1) for feat in features]

def split_shape(image):
    if isinstance(image, list):
        return [get_image_size_or_shape(im) for im in image]
    else:
        result = get_image_size_or_shape(image)
        return result

def get_image_size_or_shape(im):
    if hasattr(im, 'size'):
        size = im.size() if callable(im.size) else im.size
        return size
    else:
        shape = im.shape() if callable(im.shape) else im.shape
        return shape

def move_to_device(tensor_list, device, dtype=torch.float16):
    """Enhanced version of move_to_device that handles nested lists and ensures consistent device/dtype"""
    if isinstance(tensor_list, (list, tuple)):
        return [move_to_device(item, device, dtype) for item in tensor_list]
    elif isinstance(tensor_list, torch.Tensor):
        if tensor_list.device != device or tensor_list.dtype != dtype:
            return tensor_list.to(device=device, dtype=dtype)
        return tensor_list
    else:
        return tensor_list


class MultiVisionTower(nn.Module):
    def __init__(self, vision_towers, args, split_shape, delay_load=False):
        super().__init__()
        
        self.tile_count = split_shape[0]*split_shape[1]

        assert len(vision_towers) == self.tile_count, (
            f'Expected exactly {self.tile_count} vision towers '
            f'for a {split_shape[0]}x{split_shape[1]} split. (Currently {len(vision_towers)})'
        )
        self.vision_tower_name = "Split_Vision_Encoder"

        ######## review # Initialize vision towers for each quadrant
        self.vision_tower_list = nn.ModuleList(vision_towers)
        self.hidden_size = 0
        self.split_shape = split_shape
        if not delay_load:
            self.load_model()
        else:
            pass

        self.is_loaded = False if delay_load else True

    def load_model(self):
        for vt in self.vision_tower_list:
            vt.to(self.device)
            vt.load_model()
            self.hidden_size += vt.hidden_size
        self.is_loaded = True
        self.image_processor = [v.image_processor for v in self.vision_tower_list]


    def feature_select(self, image_forward_outs, idx):
        return self.vision_tower_list[idx].feature_select(image_forward_outs)


    def forward(self, images):
        image_features = []
        

        if isinstance(images, list): 

            for image_parts in images:

                assert len(image_parts) == self.tile_count, f'Expected exactly {self.tile_count} image parts for a {self.split_shape[0]}x{self.split_shape[1]} split. (Currently {len(image_parts)})'


                if not all(tensor.device == self.device and tensor.dtype == torch.float16 for tensor in image_parts if isinstance(tensor, torch.Tensor)):
                    image_parts = move_to_device(image_parts, self.device, torch.float16)
                part_features = []

                for idx, image_part in enumerate(image_parts):
                    image_part = image_part.to(self.device, dtype=torch.float16)
                    image_forward_outs = self.vision_tower_list[idx](image_part)
                    if not isinstance(self.vision_tower_list[idx], CLIPVisionTower):
                        image_feature = self.feature_select(image_forward_outs, idx).to(image_part.dtype)
                    part_features.append(image_feature)


                #for i in range(self.tile_count):
                #    print(f'Pre-pooling shape {i}: {part_features[i].shape}')

                target_spatial_size = 730
                part_features = adaptive_pool_features(part_features, target_spatial_size)

                #for i in range(self.tile_count):
                #    print(f'Post-pooling shape {i}: {part_features[i].shape}')
                
                image_feature_full = torch.cat(part_features, dim=2)
                image_features.append(image_feature_full) 

            image_features = torch.stack(image_features, dim=0)
            image_features = image_features.squeeze(1)

        else:  # Handling a single image
            image_parts = images
            part_features = []

            for idx, image_part in enumerate(image_parts):
                image_part = image_part.to(self.device, dtype=torch.float16)
                image_forward_outs = self.vision_tower_list[idx](image_part)
                image_feature = self.feature_select(image_forward_outs, idx).to(image_part.dtype)
                part_features.append(image_feature)

            #target_spatial_size = 730  # Set a common spatial size
            #part_features = adaptive_pool_features(part_features, target_spatial_size)

            image_features = torch.cat(part_features, dim=2)

        return image_features


    @property
    def dtype(self):
        return self.vision_tower_list[0].dtype

    @property
    def device(self):
        return self.vision_tower_list[0].device
