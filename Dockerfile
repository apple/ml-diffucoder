FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04
# `tzdata` requires noninteractive mode.

# `tzdata` requires noninteractive mode.
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG APP_DIR="/app"
ARG WANDB_API_KEY
ARG WANDB_ENTITY
ARG HUGGINGFACE_HUB_TOKEN
ARG E2B_API_KEY

# Add arguments for user and group IDs with default values
ARG USER_ID=1009
ARG GROUP_ID=1009

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Tokyo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    DEB_PYTHON_INSTALL_LAYOUT=deb \
    HOME=/home/appuser \
    PYTHONPATH=${APP_DIR}/src:$PYTHONPATH \
    WANDB_CACHE_DIR=$CACHE_DIR:wandb \
    WANDB_DATA_DIR=$CACHE_DIR:data \
    HF_HOME=$CACHE_DIR:transformer \
    HF_DATASETS_CACHE=$CACHE_DIR:datasets \
    HYDRA_FULL_ERROR=1

# Install basic dependencies and Node.js 18.x from NodeSource repository
RUN apt update \
    && apt install -y --no-install-recommends \
       curl \
       wget \
       git \
       ca-certificates \
       gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_18.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt update \
    && apt install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group with specified UID and GID
RUN groupadd -g ${GROUP_ID} appgroup \
 && useradd -u ${USER_ID} -g appgroup -m appuser \
 && chown -R appuser:appgroup /home/appuser \
 && mkdir -p /app \
 && chown -R appuser:appgroup /app \
 && mkdir -p ./.cache \
 && chmod -R 777 ./.cache

# Set the working directory
WORKDIR ${APP_DIR}

# Install npm packages as root before switching to appuser
RUN npm install -g @anthropic-ai/claude-code

# Switch to the non-root user
USER appuser

# Install conda
RUN curl -LsSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
    && bash miniconda.sh -b -p /home/appuser/miniconda3 \
    && rm miniconda.sh \
    && /home/appuser/miniconda3/bin/conda init bash

ENV PATH="/home/appuser/miniconda3/bin:$PATH"

COPY --chown=appuser:appgroup . .

CMD ["/bin/bash"]