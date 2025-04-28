#!/usr/bin/env bash

RECOMMENDED_PYTHON_VER="3.12"
PYTHON=$(basename $(command -v python${RECOMMENDED_PYTHON_VER} python3 | head -1) 2>/dev/null)

case "${PYTHON}" in
    python${RECOMMENDED_PYTHON_VER})
        echo "Python ${RECOMMENDED_PYTHON_VER} is installed, proceeding with python${RECOMMENDED_PYTHON_VER}..."
        ;;
    python3)
        echo "recommended version of Python to run exo with is Python ${RECOMMENDED_PYTHON_VER}, but $(python3 --version) is installed. Proceeding with $(python3 --version)"
        ;;
    *)
        echo "Python ${RECOMMENDED_PYTHON_VER} is not installed, please install Python ${RECOMMENDED_PYTHON_VER}"
        exit 1
        ;;
esac

${PYTHON} -m venv .venv
source .venv/bin/activate
pip install -e .
