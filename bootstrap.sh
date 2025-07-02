#!/bin/bash

MILVUS_COMPOSE="milvus-standalone.yml"
docker compose -f "$MILVUS_COMPOSE" down --remove-orphans

wget https://github.com/milvus-io/milvus/releases/download/v2.6.0-rc1/milvus-standalone-docker-compose.yml -O "$MILVUS_COMPOSE"
## Use awk to find the environment section of the standalone service and add the line
if ! grep -q "MILVUSAI_OPENAI_API_KEY:" milvus-standalone.yml; then
  awk '{
    print $0;
    if ($0 ~ /standalone:/) { in_standalone = 1; }
    if (in_standalone && $0 ~ /environment:/) {
      print "      MILVUSAI_OPENAI_API_KEY: ${OPENAI_API_KEY}";
    }
  }' milvus-standalone.yml >milvus-standalone.yml.tmp && mv milvus-standalone.yml.tmp milvus-standalone.yml
  echo "Added MILVUSAI_OPENAI_API_KEY to milvus-standalone.yml"
else
  echo "MILVUSAI_OPENAI_API_KEY already exists in milvus-standalone.yml"
fi
docker compose -f "$MILVUS_COMPOSE" up -d
