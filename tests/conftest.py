import os

# Default test secret when JWT is used during tests.
os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-jwt-secret-key-for-pytest-only-32chars",
)
