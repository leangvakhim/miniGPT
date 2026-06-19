# 1. Start with a standard Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy your requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. Copy all your code and model weights into the container
COPY . .

# 5. Start the FastAPI server on Hugging Face's required port (7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]