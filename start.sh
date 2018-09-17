#!/usr/bin/env bash
set -xeuo pipefail
IFS=$'\n\t'

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "${DIR}/generated"

docker image prune --force

docker build --build-arg USER_ID="$(id -u)" \
             --build-arg USER_GID="$(id -g)" \
             --build-arg WITH_WAND='' \
             --tag diploma_generator \
             "${DIR}"

docker run --rm -it --name diploma_generator \
           -v "${DIR}/generated":/app/generated \
           diploma_generator
