FROM python:3.11.3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV APP_HOME=/root

COPY uv.lock $APP_HOME/
COPY pyproject.toml $APP_HOME/
COPY .streamlit $APP_HOME/.streamlit
COPY app $APP_HOME/app


WORKDIR $APP_HOME
RUN uv sync

RUN uv run huggingface-cli download Alibaba-NLP/gte-modernbert-base --exclude "*onnx*"

EXPOSE 8080
CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.address", "0.0.0.0", "--server.port", "8080"]
