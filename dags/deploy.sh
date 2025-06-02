#!/bin/bash
cd /home/jyd/airflow-docker || exit 1
git reset --hard HEAD
git pull origin main
docker-compose down
docker-compose up -d
