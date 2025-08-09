def __init__(self, key: bytes = None):
    self.key = key or os.urandom(32)  # 256-bit key
    self.backend = default_backend()
    self.block_size = algorithms.AES.block_size

def encrypt_value(self, value: str) -> str:
    iv = os.urandom(16)  # Random 16-byte IV
    cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
    encryptor = cipher.encryptor()
    padded_data = self._pad(value.encode('utf-8'))
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(iv + encrypted).decode('utf-8')




def _pad(self, data: bytes) -> bytes:
    padder = padding.PKCS7(self.block_size).padder()
    return padder.update(data) + padder.finalize()

def encrypt_dataframe(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_enc = df.copy()
    for col in columns:
        df_enc[col] = df_enc[col].astype(str).apply(self.encrypt_value)
    return df_enc



def decrypt_value(self, encrypted_value: str) -> str:
    raw = base64.b64decode(encrypted_value)
    iv = raw[:16]
    encrypted = raw[16:]
    cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted) + decryptor.finalize()
    return self._unpad(padded_data).decode('utf-8')

def decrypt_dataframe(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_dec = df.copy()
    for col in columns:
        df_dec[col] = df_dec[col].apply(self.decrypt_value)
    return df_dec


def anonymize(self, df: pd.DataFrame, quasi_identifiers: list):
    grouped = df.groupby(quasi_identifiers)
    sizes = grouped.size().reset_index(name='count')
    mask = sizes['count'] >= self.k
    valid_groups = sizes[mask][quasi_identifiers]
    merged = df.merge(valid_groups, on=quasi_identifiers, how='inner')
    return merged


class KAnonymitySystem:
    def __init__(self, k=5):
        self.k = k

    def anonymize(self, df: pd.DataFrame, quasi_identifiers: list):
        if not quasi_identifiers or not all(col in df.columns for col in quasi_identifiers):
            return pd.DataFrame(columns=df.columns)
        grouped = df.groupby(quasi_identifiers)
        sizes = grouped.size().reset_index(name='count')
        mask = sizes['count'] >= self.k
        valid_groups = sizes[mask][quasi_identifiers]
        if valid_groups.empty:
            return pd.DataFrame(columns=df.columns)
        merged = df.merge(valid_groups, on=quasi_identifiers, how='inner')
        return merged

def run_performance_test(self, num_iterations=3):
    results = []
    for i in range(num_iterations):
        df = pd.read_csv(self.csv_path).head(self.num_rows)
        ksys = KAnonymitySystem(k=self.k)
        start_time = time.time()
        anonymized_df = ksys.anonymize(df, self.quasi_identifiers)
        latency = time.time() - start_time
        cpu = psutil.cpu_percent(interval=0.1)
        utility = 100.0 * len(anonymized_df) / len(df) if len(df) > 0 else 0.0
        results.append({
            "iteration": i+1, "latency": latency, "cpu_usage": cpu,
            "utility_percent": utility, "rows_retained": len(anonymized_df),
            "rows_original": len(df)
        })
    return results


