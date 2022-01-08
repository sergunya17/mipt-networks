FROM python:3.8-slim
COPY . /nn-app
WORKDIR /nn-app
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "./app.py" ]