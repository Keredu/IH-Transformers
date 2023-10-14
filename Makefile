all:
	docker build -t picasso-pytorch-image .
	docker run -it picasso-pytorch-image
