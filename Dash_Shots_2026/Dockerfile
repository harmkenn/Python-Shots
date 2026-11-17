FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/openshift/applications/ai-lab/dash-fires

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# OpenShift runs containers with an arbitrary UID. Keep app files readable
# and make the application directory group-writable for that runtime user.
RUN chgrp -R 0 /opt/openshift/applications/ai-lab/dash-fires && chmod -R g=u /opt/openshift/applications/ai-lab/dash-fires

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind=0.0.0.0:8080", "--workers=2", "--threads=4", "--timeout=120", "app:server"]
