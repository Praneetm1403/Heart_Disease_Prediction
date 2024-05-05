FROM python:3.12
ADD . /app
WORKDIR /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 5002
ENV NAME OpentoAll
CMD ["python","app.py"]