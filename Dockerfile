FROM mambaorg/micromamba:1.5.8

WORKDIR /app
COPY environment-requirements.yml /app/
COPY requirements.txt /app/

# Create env
RUN micromamba create -y -n asl -f environment-requirements.yml && \
    micromamba run -n asl pip install -r requirements.txt && \
    micromamba clean --all --yes

# Copy project
COPY . /app

# Entrypoint: run everything with default config
ENTRYPOINT ["/bin/bash", "-lc", "micromamba run -n asl ./scripts/run_all.sh config/default.yaml"]
