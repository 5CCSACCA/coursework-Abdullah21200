The Prediction Service is a key component of the Stock Prediction SaaS application. It provides authenticated RESTful endpoints for retrieving future stock price forecasts. After logging in and obtaining a session cookie, users can request predictions for the next 1 to 10 days. Internally, this service loads the latest fine-tuned LSTM model and associated scaler from a shared persistent volume, applies the model to recent scaled stock data, and returns the resulting forecasts as JSON

Key Features:

Authenticated Endpoints: Users must login firstly, create a session cookie, and can then access the /predict endpoint.
Configurable Forecast Horizon: Request between 1 to 10 days of predictions.
Scalable Deployment: The service runs as a Docker container and can be replicated on Kubernetes for improved availability.
Seamless Integration with Finetuning Service: The Prediction Service reads updated model artifacts and scaled data from a kubernetes shared volume where the Finetuning Service stores them.
