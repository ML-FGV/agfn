FROM posit/r-base:4.4-noble

ENTRYPOINT ["/bin/bash"]

# Install Python 3.10
RUN apt-get update && \
    apt-get install -y build-essential

RUN apt-get install -y python3 python3-pip python3-venv

RUN python3 -m venv /opt/venv

RUN /opt/venv/bin/python3 -m pip install --upgrade pip && \
    /opt/venv/bin/python3 -m pip install rpy2

# To watch the changes within the working directory
ARG CACHEBUST
RUN echo $CACHEBUST

COPY . .

RUN /opt/venv/bin/python3 agfn/rpy2_utils.py