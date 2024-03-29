# leverage the renci python base image
FROM renciorg/renci-python-image:v0.0.1

#Build from this branch.  Assume main for this repo
ARG BRANCH_NAME=main

# update the container
RUN apt-get update

# make a directory for the repo
RUN mkdir /repo

# go to the directory where we are going to upload the repo
WORKDIR /repo

# get the latest code
RUN git clone --branch $BRANCH_NAME --single-branch https://github.com/RobokopU24/kg-summarizer.git

# go to the repo dir
WORKDIR /repo/kg-summarizer

RUN chmod 777 -R .

# install all required packages
RUN pip install -r requirements.txt

RUN pip install uvicorn

# switch to the non-root user (nru). defined in the base image
USER nru

# expose the default port
EXPOSE 6380

# start the service entry point
ENTRYPOINT ["python", "main.py"]
