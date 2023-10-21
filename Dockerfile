# Use lightweight Python 3.8 image.
# https://hub.docker.com/_/python
FROM python:3.8-slim

# Copy local code to the container image.
ENV APP_HOME /APP_HOME
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt


# Run the web service on container startup. here, we use the gunicorn
# webserver, with one worker process and 8 threads.
CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=1", "--threads=8", "--timeout=0", "app:app"]