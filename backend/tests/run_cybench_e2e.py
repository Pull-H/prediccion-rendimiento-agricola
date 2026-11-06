"""Run the live CY-Bench API workflow with a temporary, then deactivated account."""

import argparse
import secrets
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("--validation", choices=("temporal", "parcel"), default="temporal")
parser.add_argument("--reuse-dataset-id", type=int)
args = parser.parse_args()

email = f"cybench-{secrets.token_hex(8)}@example.test"
password = secrets.token_urlsafe(24)
create = """
import sys
from app.db import SessionLocal
from app.models import User
from app.security import hasher
email = sys.stdin.readline().strip()
password = sys.stdin.readline().strip()
with SessionLocal() as db:
    db.add(User(email=email, name='CY-Bench test', role='admin', password_hash=hasher.hash(password)))
    db.commit()
"""
disable = """
import sys
from sqlalchemy import select
from app.db import SessionLocal
from app.models import User
email = sys.stdin.readline().strip()
with SessionLocal() as db:
    user = db.scalar(select(User).where(User.email == email))
    if user:
        user.active = False
        db.commit()
"""
compose = ["docker", "compose", "exec", "-T", "backend", "python", "-c"]
subprocess.run(compose + [create], input=f"{email}\n{password}\n", text=True, check=True, timeout=60)
try:
    command = [sys.executable, "backend/tests/run_cybench_docker.py", email, str(args.source),
               "--validation", args.validation]
    if args.reuse_dataset_id:
        command.extend(["--reuse-dataset-id", str(args.reuse_dataset_id)])
    subprocess.run(command, input=password + "\n", text=True, check=True, timeout=1200)
finally:
    subprocess.run(compose + [disable], input=email + "\n", text=True, check=True, timeout=60)
