# docker build -t angadsingh/argos:latest -f ../Dockerfile .

FROM argos-base:armv7

WORKDIR /usr/src/argos

COPY ./ /usr/src/argos/

EXPOSE 8081
VOLUME /output_detections
VOLUME /upload
VOLUME /configs
VOLUME /root/.ssh

EXPOSE 8080
EXPOSE 8081

ENV PYTHONPATH "${PYTHONPATH}:/configs"

ENTRYPOINT ["python3"]