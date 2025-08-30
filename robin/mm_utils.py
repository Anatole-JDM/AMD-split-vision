from PIL import Image
from io import BytesIO
import base64
import random
import numpy as np
from robin.model.multimodal_encoder.split_vision_tower import split_image, split_shape
import torch
from transformers import StoppingCriteria
from robin.constants import IMAGE_TOKEN_INDEX

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, image_aspect_ratio):
    new_images = []

    #Hardcoded because reasons.
    default_image_mean = (0.48145466, 0.4578275, 0.40821073)
    if image_aspect_ratio == 'pad':
        for image in images:
            if isinstance(image_processor, list):
                if len(image_processor) == 9:
                    split = [3,3]
                    include_full = False
                elif len(image_processor) == 10:
                    split = [3,3]
                    include_full = True
                elif len(image_processor) == 6:
                    split = [3,2]
                    include_full = False
                elif len(image_processor) == 4:
                    split = [2,2]
                    include_full = False
                elif len(image_processor) == 5:
                    split = [2,2]
                    include_full = True
                image_parts = split_image(image, split, include_full)  # Split image into parts for each vision tower
                processed_parts = []
                
                for proc, part in zip(image_processor, image_parts):
                    image_mean = getattr(proc, "image_mean", default_image_mean)
                    part = expand2square(part, tuple(int(x*255) for x in image_mean))
                    if hasattr(proc, "preprocess"):
                        processed_parts.append(proc.preprocess(part, return_tensors='pt')['pixel_values'][0])
                    else:
                        processed_parts.append(proc(part).unsqueeze(0))
                new_images.append(processed_parts) 

            else:
                # TODO: Simon: don't hardcode image mean, also this is duplicated code with train.py
                image_mean = getattr(image_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073))
                image = expand2square(image, tuple(int(x*255) for x in image_mean))

                # TODO: Simon this is nasty, we need a more unified interface here
                if hasattr(image_processor, "preprocess"):
                    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = image_processor(image).unsqueeze(0)

                new_images.append(image)
    else:
        if isinstance(image_processor, list):
            new_images = [[] for _ in range(len(image_processor))]  # Prepare list for each vision tower

            if len(image_processor) == 9:
                split = [3,3]
                include_full = False
            elif len(image_processor) == 10:
                split = [3,3]
                include_full = True
            elif len(image_processor) == 6:
                split = [3,2]
                include_full = False
            elif len(image_processor) == 4:
                split = [2,2]
                include_full = False
            elif len(image_processor) == 5:
                split = [2,2]
                include_full = True
                
            for image in images:
                image_parts = split_image(image, split, include_full)
                
                for i, (proc, part) in enumerate(zip(image_processor, image_parts)):
                    if hasattr(proc, "preprocess"):
                        new_images[i].append(proc.preprocess(part, return_tensors='pt')['pixel_values'][0])
                    else:
                        new_images[i].append(proc(part).unsqueeze(0))

            # Stack each vision tower's processed images into tensors
            return [torch.stack(parts, dim=0) for parts in new_images]

        else:
            return image_processor(images, return_tensors='pt')['pixel_values']

    if all(split_shape(x) == split_shape(new_images[0]) for x in new_images):
        if not isinstance(new_images[0], list):
            new_images = torch.stack(new_images, dim=0)
    
    return new_images


def process_images_easy(images, image_processor, image_aspect_ratio):
    new_images = []
    
    image_mean = (0.48145466, 0.4578275, 0.40821073)
    if image_aspect_ratio == 'pad':
        for image in images:

            image_mean = getattr(image_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073))
            image = expand2square(image, tuple(int(x*255) for x in image_mean))

            if hasattr(image_processor, "preprocess"):
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = image_processor(image).unsqueeze(0)

            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
        
    return new_images

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))

        self.keyword_ids = [keyword_id.to(input_ids.device) for keyword_id in self.keyword_ids]

        self.tokenizer = tokenizer
        self.batch_size = input_ids.shape[0]

        # in batch generation, is used to ensure that all samples have reached the stopping criteria
        self.matches = torch.zeros(self.batch_size, dtype=torch.int, device=input_ids.device)

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        matches = torch.zeros(self.batch_size, dtype=torch.bool, device=output_ids.device)

        for keyword_id in self.keyword_ids:
            matches |= (output_ids[:, -keyword_id.shape[0]:] == keyword_id).all(dim=1)

        for i, local_match, global_match in zip(range(self.batch_size), matches, self.matches):
            if not global_match and local_match:
                self.matches[i] = output_ids.shape[1]
                
        return self.matches.all().item()
