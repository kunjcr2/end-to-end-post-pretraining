# Password Hashing Utilities — bcrypt via passlib
#
# IMPORTANT: Always use hash_password() — NEVER Python's built-in hash().
#   hash()           → non-cryptographic integer (e.g. -1234567890), NOT secure
#   hash_password()  → bcrypt string (e.g. "$2b$12$..."), salted + secure
#
# Verification uses pwd_context.verify(), which extracts the salt from the
# stored hash and re-hashes the candidate password to compare. Direct string
# comparison (hash_password(x) == stored) does NOT work because bcrypt
# generates a unique salt each time.

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    """Hash a plaintext password with bcrypt. Returns a salted hash string."""
    return pwd_context.hash(password)

def verify_password(password: str, hashed_password: str):
    """Check a plaintext password against a bcrypt hash.
    Uses pwd_context.verify() which handles salt extraction internally.
    """
    return pwd_context.verify(password, hashed_password)