FROM python:3.11-slim

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV ENVIRONMENT=production
ENV EVAL_ENDPOINT=https://styx-evaluate-16250094868.us-central1.run.app/evaluate
ENV PROJECT_ID=16250094868

ENV PORT=8080
EXPOSE ${PORT}
#
CMD exec fastapi run main.py --port ${PORT}
