#!/bin/bash
set -e

echo "[INFO] Git pull & Docker redeploy 시작..."

cd /home/jyd/airflow-docker

# GitHub로부터 최신 코드 pull (선택사항, GitHub Actions에서 Docker Hub로 푸시했다면 불필요할 수도 있음)
# git pull origin main

# 기존 컨테이너 종료 및 삭제
docker-compose down

# 최신 이미지 pull
docker pull <DOCKER_USERNAME>/airflow-mlops:latest

# 재시작
docker-compose up -d

echo "[INFO] 배포 완료!"
