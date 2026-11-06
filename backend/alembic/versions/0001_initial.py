"""Initial relational schema.

Revision ID: 0001_initial
Revises:
"""
from alembic import op
from app.db import Base
from app import models  # noqa: F401

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    Base.metadata.create_all(bind=op.get_bind())
    if op.get_bind().dialect.name == "postgresql":
        op.execute("ALTER TABLE parcels ADD COLUMN geom geometry(Point, 4326) GENERATED ALWAYS AS (CASE WHEN latitude IS NULL OR longitude IS NULL THEN NULL ELSE ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) END) STORED")
        op.execute("CREATE INDEX ix_parcels_geom ON parcels USING GIST (geom)")

def downgrade():
    Base.metadata.drop_all(bind=op.get_bind())
