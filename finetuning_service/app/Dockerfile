# lightweight python image
FROM python:3.10-slim

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# copy source code into the container
COPY . .

# By default should run the main script monthly
CMD ["python", "main.py"]


# this all works when I run the container locally