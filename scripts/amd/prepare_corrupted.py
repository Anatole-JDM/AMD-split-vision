import shutil
import json
import random
from tqdm import tqdm

scratch = '/work1/aroger/alexisroger'

p = 50
d = 15
n = 3

for i in tqdm(range(n)):
    name = f'finetune_{p}_{d}_{i}'
    dst = f'{scratch}/checkpoints/robin_full_test/{name}'
    
    base = f'{scratch}/checkpoints/robin_full_test/finetune_base'
    shutil.copytree(base, dst)

    data_path = f'{dst}/train_monkey_densecap.json'
    shutil.copy2(f'{scratch}/data/train_monkey_densecap_2.json', data_path)


    with open(data_path) as f_data:
        data = json.load(f_data)
        N = len(data)-1
        to_corrupt = random.sample(range(N), N*d//100)

    with open(f'{scratch}/data/corrupted-data/monkey_densecap_{p}%_chance_mask_gemini.json') as f_corrupted:
        corrupted = json.load(f_corrupted)
        for j in to_corrupt:
            try:
                data[j]['conversations'][1]['value'] = corrupted[j]['conversations'][1]['value']
            except:
                print(j)
                print(data[j])
                print(corrupted[j])
                print()
                raise

        with open(data_path, 'w') as f_data:
            json.dump(data, f_data)
