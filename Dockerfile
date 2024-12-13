FROM python:3.9-slim
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --proxy http://proxy365.sacombank.com:1985 -r requirements.txt
COPY . /opt/
WORKDIR /opt
EXPOSE 5000
CMD ["python", "app.py"]