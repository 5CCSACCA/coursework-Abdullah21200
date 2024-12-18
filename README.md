[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pa_hoUiU)

# firsly make sure youre in the correct directory when running,build or testing 

# create image in finetuning/app directory

    docker build -t finetuning_service_image:latest .
    docker build -t prediction_service_image:latest .

# create a cluster called mycluster
    kind create cluster --name predictioncluster --config kind_config.yaml 

#its important you go into this file and put abs path of <shared_model>
                                                                                    

# add finetuning image (or any other image) to cluster
    kind load docker-image finetuning_service_image:latest --name predictioncluster
    kind load docker-image prediction_service_image:latest --name predictioncluster


# apply these configs to mount model *Important*
    kubectl apply -f shared_storage_pv.yaml
    kubectl apply -f shared_storage_pvc.yaml
    kubectl apply -f finetuning_job.yaml
    kubectl apply -f prediction_deployment.yaml

# port forward to access the service
    kubectl port-forward service/prediction-service 10000:8000

# login to create a session
    curl -X POST "http://localhost:30000/login" -u user1:xxxxxx -i

# copy set cookies data from the response and paste it in the next curl command
    curl -X POST "http://localhost:30000/predict?days=5" -H "Cookie: <ccopied_cookie-data>" 