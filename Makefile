help:
	@echo 'make build -> to build'
	@echo 'make run -> to run'
	@echo 'make clean -> to stop jupyter container and delete it'
build:
	docker build -t jupyter deployment
run:
	docker run --name ml_env -d -p 8888:8888 -v $(PWD)/notebooks:/home/jupyter jupyter
	docker logs ml_env
clean:
	docker stop ml_env
	docker rm ml_env
	docker rmi jupyter
