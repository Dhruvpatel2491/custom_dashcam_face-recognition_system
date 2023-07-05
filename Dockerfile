FROM animcogn/face_recognition

# for Local Uncomment Below
# RUN mkdir original_image
# COPY original_image original_image/
# COPY local/encodeGenerator_local.py .

#for AWS
COPY aws/encodeGenerator_aws.py .

RUN pip install --upgrade pip
RUN apt-get -y update
RUN apt-get install -y python3.7
RUN pip3 install numpy
RUN pip3 install opencv-python-headless

# Local RUN
# CMD ["python3","encodeGenerator_local.py"]

# For AWS Docker Image
CMD ["python3","aws/encodeGenerator_aws.py"]

