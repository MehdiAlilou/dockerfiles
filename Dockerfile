FROM pytorch/pytorch:latest
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python ./maskrcnn.py ./input_output/