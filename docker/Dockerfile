FROM tensorflow/tensorflow:2.16.2-gpu
RUN pip install scikit-image==0.24.0
RUN pip install scikit-video==1.1.11
RUN pip install matplotlib==3.9.1
RUN pip install -I mlflow
RUN pip install deepvog==1.2.1
RUN apt update -y
RUN apt install -y ffmpeg