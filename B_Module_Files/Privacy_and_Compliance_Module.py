"""
Privacy_and_Compliance_Module.py

Absolute File Path (example):
    /Users/username/Desktop/energy_optimization_project/B_Module_Files/Privacy_and_Compliance_Module.py

PURPOSE:
  - Handles data encryption/decryption.
  - Provides anonymization or redaction placeholders for sensitive fields.
  - Includes stub methods for GDPR/CCPA compliance requests (export, erasure).
  - Potential placeholder for differential privacy or other advanced privacy tech.

NOTES:
  - In production, ensure consistent key management; do not generate ephemeral keys for real data.
  - This module can be extended to handle data retention policies, secure data shredding, etc.
"""

import logging
import os
import base64
from typing import Union, List, Dict, Any
from cryptography.fernet import Fernet


class PrivacyAndCompliance:
    """
    A class providing data privacy and compliance functionalities:
      - Encryption/decryption of sensitive data using Fernet (symmetric encryption).
      - Anonymization placeholders to remove or mask PII (personally identifiable info).
      - Stub methods to handle GDPR-like requests (data export, data erasure, etc.).
      - Optional placeholders for differential privacy or advanced compliance checks.
    """

    def __init__(self, encryption_key: str = ""):
        """
        :param encryption_key: A 32-byte url-safe base64-encoded key string.
                              If empty, attempts to load from ENV: ENCRYPTION_KEY
                              If none found, a random key is generated (not secure for real usage).
        """
        self.logger = logging.getLogger("PrivacyAndCompliance")
        if encryption_key:
            self.key = encryption_key
            self.logger.debug("Encryption key provided directly.")
        else:
            # Generate or fallback to environment variable
            env_key = os.getenv("ENCRYPTION_KEY", "")
            if env_key:
                self.key = env_key
                self.logger.debug("Encryption key loaded from ENCRYPTION_KEY env variable.")
            else:
                # Last resort: generate a new key (ephemeral)
                random_key = Fernet.generate_key()
                self.key = random_key.decode("utf-8")
                self.logger.warning("No encryption key found; generated a temporary key (not persistent).")

        # Ensure we have a valid key (32 url-safe base64-encoded bytes).
        # If user provides a string shorter than 32 chars, we "pad" it out to 32 with '='
        if len(self.key) < 32:
            # Convert whatever was provided to a URL-safe base64. Then pad if needed.
            enc = base64.urlsafe_b64encode(self.key.encode("utf-8"))
            self.key = enc.ljust(32, b'=').decode("utf-8")

        self.fernet = Fernet(self.key.encode("utf-8"))
        self.logger.info("PrivacyAndCompliance module initialized with a valid encryption mechanism.")

    def encrypt_data(self, plaintext: Union[str, bytes]) -> str:
        """
        Encrypt plaintext data into a Fernet token (string).
        :param plaintext: either str or bytes data to encrypt
        :return: base64-encoded ciphertext as a string
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")
        token = self.fernet.encrypt(plaintext)
        self.logger.debug("Data successfully encrypted.")
        return token.decode("utf-8")

    def decrypt_data(self, token: Union[str, bytes]) -> str:
        """
        Decrypt a Fernet token back into a string.
        :param token: either str or bytes ciphertext token
        :return: decrypted string
        """
        if isinstance(token, str):
            token = token.encode("utf-8")
        decrypted = self.fernet.decrypt(token)
        self.logger.debug("Data successfully decrypted.")
        return decrypted.decode("utf-8")

    def anonymize_record(self, record: Dict[str, Any], fields_to_remove: List[str] = None) -> Dict[str, Any]:
        """
        Remove or mask PII fields from a record.
        :param record: a dictionary representing a record with potential PII
        :param fields_to_remove: list of keys to redact. Default: ["name", "email", "phone"]
        :return: a new record dict with sensitive fields replaced by "REDACTED"
        """
        if fields_to_remove is None:
            fields_to_remove = ["name", "email", "phone"]
        new_rec = record.copy()
        for f in fields_to_remove:
            if f in new_rec:
                new_rec[f] = "REDACTED"
        self.logger.debug(f"Record anonymized. Fields {fields_to_remove} were redacted.")
        return new_rec

    def gdpr_export_data(self, user_id: str) -> Dict[str, Any]:
        """
        Stub method for GDPR 'right to data portability' or export request.
        In reality, you'd fetch all user data from your DB or logs and return.
        :param user_id: ID or unique identifier of the user requesting data
        :return: dictionary with user data
        """
        self.logger.info(f"Performing GDPR data export for user {user_id} (prototype stub).")
        # In a real scenario, query user data from DB, logs, etc. Here we simulate a response:
        return {
            "user_id": user_id,
            "export_date": "2024-01-01",
            "profile_data": {
                "name": "John Doe",
                "preferences": {"lang": "EN", "timezone": "UTC"}
            }
        }

    def gdpr_delete_data(self, user_id: str) -> bool:
        """
        Stub method for GDPR 'right to be forgotten' or data erasure request.
        In reality, you'd remove user data from your DB.
        :param user_id: ID or unique identifier of the user requesting deletion
        :return: True if successful, else False
        """
        self.logger.info(f"Performing GDPR data deletion for user {user_id} (prototype stub).")
        # Real scenario: delete user data from DB, logs, etc.
        # Return True on success, False otherwise
        return True

    def differential_privacy_mechanism(self, data: float, epsilon: float = 1.0) -> float:
        """
        Placeholder for applying differential privacy noise to a numeric data point.
        e.g., Laplace mechanism. 
        :param data: original numeric data
        :param epsilon: privacy budget parameter
        :return: data with noise added
        """
        self.logger.info(f"Applying differential privacy with epsilon={epsilon} (placeholder).")
        # Very naive approach: add Laplace noise with scale = 1/epsilon
        # For real usage, see e.g. PyDP library or Google DP frameworks
        import random
        import math

        scale = 1.0 / epsilon
        # Simple sample from Laplace(0, scale)
        # Laplace distribution can be simulated by sampling from uniform(0,1), transform.
        # This is a demonstration, not a robust Laplace sampler.
        u = random.random() - 0.5
        # Laplace(0,b) can be generated as -b * sgn(u) * ln(1 - 2|u|)
        noise = -scale * math.copysign(math.log(1 - 2 * abs(u)), u)
        return data + noise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pc = PrivacyAndCompliance(encryption_key="my-super-secret-key-123")

    # Encryption/Decryption example
    enc = pc.encrypt_data("Sensitive data: user password = 1234")
    print("Encrypted text:", enc)
    dec = pc.decrypt_data(enc)
    print("Decrypted text:", dec)

    # Anonymization example
    record = {"name": "Alice", "email": "alice@example.com", "temp": 23.0, "location": "EU"}
    masked = pc.anonymize_record(record, ["name", "email"])
    print("Anonymized record:", masked)

    # GDPR data export (stub)
    export_result = pc.gdpr_export_data("user_42")
    print("GDPR Export data (stub):", export_result)

    # GDPR data delete (stub)
    delete_success = pc.gdpr_delete_data("user_42")
    print("GDPR Delete success?:", delete_success)

    # Differential Privacy demonstration
    original_value = 42.0
    dp_value = pc.differential_privacy_mechanism(original_value, epsilon=1.0)
    print(f"Original: {original_value}, DP version: {dp_value}")