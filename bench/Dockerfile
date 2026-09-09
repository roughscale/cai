FROM kali-headless-base:latest

RUN apt-get update && apt-get install -y --no-install-recommends seclists && apt-get clean && rm -rf /var/lib/apt/lists/*

# seclists installs to /usr/share/seclists; symlink to where the system prompt expects them
RUN mkdir -p /usr/share/wordlists && ln -s /usr/share/seclists /usr/share/wordlists/seclists

RUN mkdir -p /root/.pip && printf '[global]\nbreak-system-packages = true\n' > /root/.pip/pip.conf

# Install CAI from source so branch changes are included
WORKDIR /opt/cai-src
COPY . .
RUN pip3 install --ignore-installed .

WORKDIR /opt/cai
RUN mkdir -p logs

ENTRYPOINT ["python3", "/opt/cai-src/bench/runner.py"]
