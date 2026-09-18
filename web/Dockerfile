# web - the core/testing service: runs src/pipeline as background jobs
# and is the UI's single API surface. Build context is the REPO ROOT
# (see docker-compose.yml), since this needs src/ alongside its own code.
FROM python:3.12-slim

# wkhtmltopdf (needed by src/pipeline/pdf_export.py) was dropped from
# Debian's own repos starting with bookworm - its patched Qt WebKit
# fork is unmaintained upstream, and Debian 11 "bullseye" (the last
# release that packaged it) is now EOL with stale security-repo
# mirrors, so apt can't install it there either. Instead, fetch the
# project's own "generic linux" static build directly (self-contained;
# doesn't need an X server since 0.12.x's headless mode) and install
# just its handful of real runtime libs (X11/font client libs) via apt
# from this (current, maintained) base's own repos.
RUN apt-get update && apt-get install -y --no-install-recommends \
        pandoc \
        libxrender1 \
        libxext6 \
        libfontconfig1 \
        fonts-dejavu-core \
        curl \
        xz-utils \
    && curl -fsSL -o /tmp/wkhtmltox.tar.xz \
        "https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz" \
    && tar -xf /tmp/wkhtmltox.tar.xz -C /opt \
    && ln -s /opt/wkhtmltox/bin/wkhtmltopdf /usr/local/bin/wkhtmltopdf \
    && ln -s /opt/wkhtmltox/bin/wkhtmltoimage /usr/local/bin/wkhtmltoimage \
    && rm /tmp/wkhtmltox.tar.xz \
    && apt-get purge -y curl xz-utils \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY web/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY web/*.py /app/

ENV PYTHONPATH=/app/src
ENV RESULTS_DIR=/app/results

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
