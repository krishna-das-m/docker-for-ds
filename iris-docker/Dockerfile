FROM python:3.11-slim

# Copy uv binary from official uv image (multi-stage build pattern)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# set working directory inside container
WORKDIR /app

# Add virtual environment to PATH so we can use installed packages
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency files first (better layer caching)
COPY "pyproject.toml" "uv.lock" ".python-version" ./
# Install dependencies from lock file (ensures reproducible builds)
RUN uv sync --locked

# # Install python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY . .

# default command to run when container starts
# CMD [ "python", "train.py"]

# Set entry point
ENTRYPOINT ["uv", "run", "python", "train_and_save.py"]
