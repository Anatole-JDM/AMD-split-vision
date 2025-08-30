import torch
import os
import json
import requests

from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from robin.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robin.conversation import conv_templates, SeparatorStyle
from robin.model.builder import load_pretrained_model
from robin.utils import disable_torch_init
from robin.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from robin.model.multimodal_encoder.split_vision_tower import move_to_device

class Robin:
    def __init__(self, 
                 model_path,
                 model_base="",
                 device="cuda",
                 conv_mode="vicuna_v1",
                 temperature=0.2,
                 top_p=None,
                 max_new_tokens=512,
                 load_8bit=False,
                 load_4bit=False,
                 debug=False,
                 image_aspect_ratio='pad',
                 lazy_load=False,
                 split_shape=None):
        
        self.model_path = os.path.expanduser(model_path)
        self.model_base = model_base
        self.device = device
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.debug = debug
        self.image_aspect_ratio = image_aspect_ratio
        self.split_shape = split_shape

        self.loaded = False
        if not lazy_load:
            self.load_model()

    def find_base(self):
        if os.path.exists(self.model_path):
            if os.path.exists(os.path.join(self.model_path, 'non_lora_trainables.bin')):
                with open(os.path.join(self.model_path, "adapter_config.json"), "r") as f:
                    config = json.load(f)
                    print("CONFIG ", config)
                    self.model_base = config["base_model_name_or_path"]
            else:
                self.model_base = None
        else:
            print("No local model found, trying to download from Hugging Face")
            try:
                print("Trying to download model from Hugging Face", self.model_path, '...')
                url = f"https://huggingface.co/{self.model_path}/raw/main/adapter_config.json"
                response = requests.get(url)
                if response.status_code != 200:
                    response = requests.get(url, headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]})
                if response.status_code != 200:
                    print("No model base found for ", self.model_path)
                    self.model_base = None
                    return
                config = json.loads(response.text)
                self.model_base = config["base_model_name_or_path"]
                print("Model base found at", self.model_base)
            except Exception as e:
                print("Failed to find base on Hugging Face", self.model_path, e)
                self.model_base = None

    def load_model(self):
        disable_torch_init()

        self.model_name = get_model_name_from_path(self.model_path)

        if self.model_base is not None:
            if len(self.model_base) == 0:
                self.find_base()
        
        print(f"Loading model {self.model_name} from {self.model_base}...")

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, self.model_base, self.model_name, self.load_8bit, self.load_4bit, device=self.device)
        
        self.loaded = True

    def load_image(self, image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def __call__(self, img_url, prompt, streamer=False):
        if not self.loaded:
            self.load_model()

        conv = conv_templates[self.conv_mode].copy()
        roles = conv.roles

        if img_url is not None:
            if isinstance(img_url, str):
                if len(img_url.strip()) == 0:
                    img_url = None

        if img_url is not None:
            if isinstance(img_url, str):
                image = self.load_image(img_url)
            else:
                image = img_url

            # Similar operation in model_worker.py
            image_tensor = process_images([image], self.image_processor, self.image_aspect_ratio)
            image_tensor = move_to_device(image_tensor, self.model.device, dtype=torch.float16)


            if self.model.config.mm_use_im_start_end:
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        else:
            image_tensor = None

        if self.debug: print(f"{roles[1]}: ", end="")

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)]
        if streamer: 
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=stopping_criteria
                )
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=stopping_criteria
                )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:])
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

        conv.messages[-1][-1] = outputs

        if self.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        return outputs
