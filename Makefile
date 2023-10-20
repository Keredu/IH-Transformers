all:
	docker build -t picasso-pytorch-image .
	docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ./experiments:/workspace/results picasso-pytorch-image

