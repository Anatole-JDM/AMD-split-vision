iter=(
        "0"
        "1"
        "2"
        "3"
        "4"
)

base="/work1/aroger/alexisroger/downloaded_models"

cd /work1/aroger/alexisroger/checkpoints/robin_full_test

git lfs install
for i in "${iter[@]}"
do
        git_name=corrupted_finetuning_50p_10d_$i
        model=finetune_50_10_$i
        
        git clone hf:MongolLabs/corrupted_finetuning_50p_10d_$i
        cp $model/* $git_name/
        cd $git_name

        rm train_monkey_densecap.json

        sed -i "s|$base/OpenHermes-2.5-Mistral-7B|teknium/OpenHermes-2.5-Mistral-7B|g" *.{json,md}
        sed -i "s|$base/ViT-SO400M-14-SigLIP-384|hf-hub:timm/ViT-SO400M-14-SigLIP-384|g" config.json
        git add *
        git commit -m model
        git push
        cd ..

        rm -rf $git_name

done