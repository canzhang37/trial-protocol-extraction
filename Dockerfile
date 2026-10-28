FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir ".[dev]"

# data/ and results/ are mounted at run time
ENTRYPOINT ["python", "-m", "protex"]
CMD ["--help"]
