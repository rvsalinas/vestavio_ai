"""
security_layer_module.py

Absolute File Path (example):
    /Users/robertsalinas/Desktop/energy_optimization_project/B_Module_Files/security_layer_module.py

PURPOSE:
  - Provides security features such as JWT-based authentication, password hashing, and data encryption.
  - Handles user authentication flows (token generation/verification) and optionally compliance placeholders.

NOTES:
  - Uses environment variables for SECRET_KEY, ENCRYPTION_KEY, etc.
  - Incorporates bcrypt for password hashing/verification.
  - Uses PyJWT for token generation/validation (JWT).
  - Implements optional encryption via Fernet (symmetric encryption).
  - This code is for demonstration and not production-ready; always follow best practices for secrets management.

EXAMPLE USAGE:
  sl = SecurityLayer(secret_key="your-jwt-secret", encryption_key="16+_char_encryption_key")
  token = sl.generate_token("user_42")  # create a JWT
  payload = sl.verify_token(token)      # decode/verify the JWT
  hashed_pw = sl.hash_password("mypassword")
  check_ok = sl.verify_password("mypassword", hashed_pw)
  secure_data = sl.encrypt_data("Some sensitive info")
  original_data = sl.decrypt_data(secure_data)
"""

import os
import time
import logging
import jwt
import bcrypt
from typing import Optional

# For encryption
from cryptography.fernet import Fernet, InvalidToken


class SecurityLayer:
    """
    A class encapsulating security functionalities:
      - JWT token generation/verification
      - Password hashing with bcrypt
      - Symmetric encryption/decryption with Fernet
      - Placeholder for compliance checks (e.g. logs, audits)
    """

    def __init__(
        self,
        secret_key: str = "",
        encryption_key: str = "",
        token_algorithm: str = "HS256",
        token_expiration_seconds: int = 3600
    ):
        """
        :param secret_key: Secret used for JWT token signing.
        :param encryption_key: 16+ char key for Fernet encryption (if not set, generates one).
        :param token_algorithm: Algorithm for JWT (usually "HS256").
        :param token_expiration_seconds: Default expiration for generated tokens.
        """
        # Setup logging
        self.logger = logging.getLogger("SecurityLayer")
        self.logger.setLevel(logging.INFO)

        # JWT secret key
        self.secret_key = secret_key or os.getenv("SECRET_KEY", "fallback_jwt_secret")
        if len(self.secret_key) < 8:
            self.logger.warning(
                "Secret key is too short; using fallback or artificially extended key."
            )
            self.secret_key = (self.secret_key + "XYZXYZ123")[:16]

        self.token_algorithm = token_algorithm
        self.default_exp_seconds = token_expiration_seconds

        # Encryption key for Fernet
        # Must be url-safe base64-encoded 32 bytes. You can store it in an env var or vault
        # If user doesn't provide, we either generate or fallback to environment
        self.encryption_key = encryption_key or os.getenv("ENCRYPTION_KEY", "")
        if not self.encryption_key:
            # Generate a new Fernet key for demonstration
            # (Do NOT do this in production if you need stable encryption across sessions)
            self.encryption_key = Fernet.generate_key().decode("utf-8")
            self.logger.warning("No encryption_key specified. Generated a new one (not persistent).")

        # Ensure encryption_key is 32 url-safe base64 chars. If user-supplied is raw text, you'd need
        # to convert to base64. We'll assume the user provides a valid base64-encoded 32 char key or does so in env.
        try:
            self.fernet = Fernet(self.encryption_key.encode("utf-8"))
        except Exception as e:
            self.logger.error(
                f"Invalid encryption_key provided. Will not perform encryption/decryption. Error: {e}"
            )
            self.fernet = None

        self.logger.info("SecurityLayer initialized successfully.")

    # ---------------------------
    # JWT Token Methods
    # ---------------------------
    def generate_token(self, user_id: str, exp_seconds: Optional[int] = None) -> str:
        """
        Create a JWT token for 'user_id'. 
        :param user_id: The user identifier to embed in the token.
        :param exp_seconds: Expiration in seconds from now. Default self.default_exp_seconds if None.
        :return: Encoded JWT token string.
        """
        if exp_seconds is None:
            exp_seconds = self.default_exp_seconds
        now = int(time.time())
        payload = {
            "id": user_id,
            "iat": now,
            "exp": now + exp_seconds
        }
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.token_algorithm)
            # In PyJWT >=2.0, encode() returns a str in some setups, bytes in older versions
            if isinstance(token, bytes):
                token = token.decode("utf-8")
            return token
        except Exception as e:
            self.logger.error(f"Error generating token: {e}", exc_info=True)
            return ""

    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify a JWT token. Return the decoded payload if valid, else None.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.token_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired.")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token.")
            return None

    # ---------------------------
    # Password Methods
    # ---------------------------
    def hash_password(self, password: str) -> str:
        """
        Hash a plaintext password using bcrypt.
        :return: The hashed password string (e.g., bcrypted).
        """
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
            return hashed.decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error hashing password: {e}", exc_info=True)
            return ""

    def verify_password(self, password: str, hashed_pw: str) -> bool:
        """
        Verify a plaintext password against a hashed password from storage.
        """
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed_pw.encode("utf-8"))
        except Exception as e:
            self.logger.error(f"Error verifying password: {e}", exc_info=True)
            return False

    # ---------------------------
    # Encryption/Decryption
    # ---------------------------
    def encrypt_data(self, plaintext: str) -> str:
        """
        Symmetric encryption of the given plaintext using Fernet.
        Returns base64-encoded ciphertext.
        """
        if not self.fernet:
            self.logger.warning("No valid Fernet key. Encryption not performed.")
            return plaintext  # fallback: return as is
        try:
            encrypted = self.fernet.encrypt(plaintext.encode("utf-8"))
            return encrypted.decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encrypting data: {e}", exc_info=True)
            return plaintext  # fallback

    def decrypt_data(self, ciphertext: str) -> str:
        """
        Decrypt ciphertext (base64-encoded from encrypt_data) using Fernet.
        Returns the original plaintext or the ciphertext if error.
        """
        if not self.fernet:
            self.logger.warning("No valid Fernet key. Decryption not performed.")
            return ciphertext
        try:
            decrypted = self.fernet.decrypt(ciphertext.encode("utf-8"))
            return decrypted.decode("utf-8")
        except (InvalidToken, Exception) as e:
            self.logger.error(f"Error decrypting data: {e}", exc_info=True)
            return ciphertext

    # ---------------------------
    # Compliance Placeholders
    # ---------------------------
    def log_compliance_event(self, event_msg: str) -> None:
        """
        Placeholder for compliance logging or auditing events.
        """
        self.logger.info(f"[COMPLIANCE LOG]: {event_msg}")

    def perform_audit_check(self) -> None:
        """
        Placeholder for performing compliance or audit checks.
        """
        self.logger.info("Performing compliance audit check (placeholder).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demonstration
    sec_layer = SecurityLayer(
        secret_key="my_secret_jwt_key",
        encryption_key="",  # Will auto-generate or fallback if not provided
        token_algorithm="HS256",
        token_expiration_seconds=60
    )

    # Token example
    user_token = sec_layer.generate_token(user_id="user42")
    print("Generated Token:", user_token)
    decoded = sec_layer.verify_token(user_token)
    print("Decoded token payload:", decoded)

    # Password example
    plain_pw = "SuperSecret123!"
    hashed_pw = sec_layer.hash_password(plain_pw)
    print("Hashed Password:", hashed_pw)
    verified = sec_layer.verify_password(plain_pw, hashed_pw)
    print("Password verified?:", verified)

    # Encryption example
    secret_message = "This is top secret."
    enc = sec_layer.encrypt_data(secret_message)
    dec = sec_layer.decrypt_data(enc)
    print("Encrypted message:", enc)
    print("Decrypted message:", dec)

    # Compliance placeholder
    sec_layer.log_compliance_event("User user42 performed an admin action.")
    sec_layer.perform_audit_check()