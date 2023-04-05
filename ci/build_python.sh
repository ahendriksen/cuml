#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

PY_VER=${RAPIDS_PY_VERSION//./}

LIBRMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1241/1fd0b84/rmm_conda_cpp_cuda11_$(arch).tar.gz)
RMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1241/1fd0b84/rmm_conda_python_cuda11__${PY_VER}_$(arch).tar.gz)

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/cuml

rapids-upload-conda-to-s3 python
