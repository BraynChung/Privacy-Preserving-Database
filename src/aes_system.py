import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os
import base64

class AESEncryptionSystem:
    """AES Encryption/Decryption for DataFrames."""

    def __init__(self, key: bytes = None):
        # AES key must be 16, 24, or 32 bytes long
        self.key = key or os.urandom(32)
        self.backend = default_backend()
        self.block_size = algorithms.AES.block_size

    def _pad(self, data: bytes) -> bytes:
        padder = padding.PKCS7(self.block_size).padder()
        return padder.update(data) + padder.finalize()

    def _unpad(self, padded_data: bytes) -> bytes:
        unpadder = padding.PKCS7(self.block_size).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()

    def encrypt_value(self, value: str) -> str:
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        padded_data = self._pad(value.encode('utf-8'))
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        return base64.b64encode(iv + encrypted).decode('utf-8')

    def decrypt_value(self, encrypted_value: str) -> str:
        raw = base64.b64decode(encrypted_value)
        iv = raw[:16]
        encrypted = raw[16:]
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted) + decryptor.finalize()
        return self._unpad(padded_data).decode('utf-8')

    def encrypt_dataframe(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_enc = df.copy()
        for col in columns:
            df_enc[col] = df_enc[col].astype(str).apply(self.encrypt_value)
        return df_enc

    def decrypt_dataframe(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_dec = df.copy()
        for col in columns:
            df_dec[col] = df_dec[col].apply(self.decrypt_value)
        return df_dec

# Example usage:
if __name__ == "__main__":
    # Load a sample of your data
    df = pd.read_csv("dataset/adult.csv").head(10)
    columns_to_encrypt = ['education', 'occupation']  # Example columns

    aes_system = AESEncryptionSystem()
    encrypted_df = aes_system.encrypt_dataframe(df, columns_to_encrypt)
    decrypted_df = aes_system.decrypt_dataframe(encrypted_df, columns_to_encrypt)

    print("Original:\n", df[columns_to_encrypt].head())
    print("Encrypted:\n", encrypted_df[columns_to_encrypt].head())
    print("Decrypted:\n", decrypted_df[columns_to_encrypt].head())