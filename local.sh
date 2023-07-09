#!/usr/bin/env bash
# Create virtualenv if not exist
export VIRTUALENV_PATH=$(pwd)/venv
if [[ -d "${VIRTUALENV_PATH}" ]]; then
  echo "Skipping creating virtualenv at ${VIRTUALENV_PATH} - already exist."
else
  echo "Creating virtualenv at ${VIRTUALENV_PATH}."
  virtualenv -p python3 "${VIRTUALENV_PATH}"
fi

# Activate virtualenv
source "${VIRTUALENV_PATH}"/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/venv/lib/python3.9/site-packages/"

# Get project directory
export APP_PATH=$(pwd)
export SRC_PATH=${SRC_PATH:-${APP_PATH}/src}
export PYTHONPATH=${PYTHONPATH}:${SRC_PATH}

export TF_FORCE_GPU_ALLOW_GROWTH="true"

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
