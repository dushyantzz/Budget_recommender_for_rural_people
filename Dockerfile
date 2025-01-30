# Use Python 3.9 as a base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code & model
COPY main.py /app/
COPY budget_recommender_for_rural_people.py /app/
COPY best_model.pkl /app/

# Expose the port (FastAPI will run on 80 inside the container)
EXPOSE 80

# Start FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
