#!/bin/bash

MILVUS_COMPOSE="milvus-standalone.yml"
docker compose -f "$MILVUS_COMPOSE" down --remove-orphans

wget https://github.com/milvus-io/milvus/releases/download/v2.6.0-rc1/milvus-standalone-docker-compose.yml -O "$MILVUS_COMPOSE"
docker compose -f "$MILVUS_COMPOSE" up -d
