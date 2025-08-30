#	-it \
docker run \
	--rm \
	--detach \
	--gpus 2 \
	--shm-size 8G \
	--volume /home/$(whoami)/checkpoints:/export \
	docker.io/library/robin_pretrain \
		teknium/OpenHermes-2.5-Mistral-7B \
		facebook/metaclip-l14-fullcc2.5b \
		2
