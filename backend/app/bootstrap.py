import argparse
import getpass
from sqlalchemy import select
from .db import Base, SessionLocal, engine
from .models import User
from .security import hasher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default="Administrador")
    args = parser.parse_args()
    password = getpass.getpass("Nueva contraseña de administrador: ").rstrip("\r\n")
    if len(password) < 12:
        raise SystemExit("La contraseña debe tener al menos 12 caracteres")
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.email == args.email)):
            raise SystemExit("El usuario ya existe")
        db.add(User(email=args.email, name=args.name, role="admin", password_hash=hasher.hash(password)))
        db.commit()
    print("Administrador creado")

if __name__ == "__main__":
    main()
