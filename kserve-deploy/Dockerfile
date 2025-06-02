#Python 3.11
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /opt/application

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY project project

CMD python -m project.serve --enable_docs_url true
