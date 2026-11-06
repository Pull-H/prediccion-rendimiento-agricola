"""Add revocable sessions to existing installations.

Revision ID: 0002_sessions
Revises: 0001_initial
"""
from alembic import op
from sqlalchemy import inspect
from app.models import SessionToken

revision = "0002_sessions"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

def upgrade():
    bind = op.get_bind()
    if not inspect(bind).has_table("session_tokens"):
        SessionToken.__table__.create(bind=bind)

def downgrade():
    SessionToken.__table__.drop(bind=op.get_bind(), checkfirst=True)
