FROM python:3.12 as nemo-run-update
WORKDIR /opt
ARG NEMO_RUN_COMMIT
RUN <<"EOF" bash -exu
if [[ ! -d NeMo-Run ]]; then
    git clone https://github.com/NVIDIA/NeMo-Run.git
fi
cd NeMo-Run
git init
git remote add origin https://github.com/NVIDIA/NeMo-Aligner.git
git fetch --all
git fetch origin '+refs/pull/*/merge:refs/remotes/pull/*/merge'
git checkout $NEMO_RUN_COMMIT
EOF

FROM python:3.12

RUN \
    --mount=type=bind,source=/opt/NeMo-Run/src/nemo_run/__init__.py,target=/opt/NeMo-Run/src/nemo_run/__init__.py,from=nemo-run-update \
    --mount=type=bind,source=/opt/NeMo-Run/src/nemo_run/package_info.py,target=/opt/NeMo-Run/src/nemo_run/package_info.py,from=nemo-run-update \
    --mount=type=bind,source=/opt/NeMo-Run/pyproject.toml,target=/opt/NeMo-Curator/pyproject.toml,from=nemo-run-update \
    cd /opt/NeMo-Run && \
    pip install .

COPY --from=nemo-run-update /opt/NeMo-Run/ /opt/NeMo-Run/

# Clone the user's repository, find the relevant commit, and install everything we need
RUN bash -exu <<EOF
  cd /opt/NeMo-Run/
  pip install .
EOF
