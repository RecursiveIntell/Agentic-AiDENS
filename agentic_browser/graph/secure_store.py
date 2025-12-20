"""
Secure Storage Layer for Agent Knowledge Banks.

Provides:
- AES-256 encryption (Fernet) for high-value databases (strategies, apocalypse)
- HMAC-SHA256 signing for tamper detection on lower-value databases (runs)
- Key management with auto-generation and secure file permissions
"""

import os
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

# Try to import cryptography, provide helpful error if missing
try:
    from cryptography.fernet import Fernet, InvalidToken
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None
    InvalidToken = Exception


class SecureStore:
    """Handles encryption, decryption, and integrity verification."""
    
    KEY_FILE = ".key"
    MEMORY_DIR = "memory"
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize secure store.
        
        Args:
            base_dir: Base directory for storage (default: ~/.agentic_browser)
        """
        self.base_dir = base_dir or Path.home() / ".agentic_browser"
        self.memory_dir = self.base_dir / self.MEMORY_DIR
        self.key_path = self.base_dir / self.KEY_FILE
        
        # Ensure directories exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "planner").mkdir(exist_ok=True)
        (self.memory_dir / "research").mkdir(exist_ok=True)
        
        # Load or generate key
        self._key: Optional[bytes] = None
        self._fernet: Optional[Any] = None
        
    @property
    def key(self) -> bytes:
        """Get or generate the master encryption key."""
        if self._key is None:
            if self.key_path.exists():
                self._key = self.key_path.read_bytes()
            else:
                self._key = self._generate_key()
        return self._key
    
    @property
    def fernet(self) -> Any:
        """Get Fernet cipher instance."""
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography package not installed. "
                "Install with: pip install cryptography"
            )
        if self._fernet is None:
            self._fernet = Fernet(self.key)
        return self._fernet
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key and save it securely."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for key generation")
            
        key = Fernet.generate_key()
        
        # Write key with restricted permissions (owner only)
        self.key_path.write_bytes(key)
        os.chmod(self.key_path, 0o600)
        
        print(f"🔐 Generated new encryption key: {self.key_path}")
        return key
    
    # ==================== ENCRYPTION ====================
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using Fernet (AES-256).
        
        Args:
            data: Raw bytes to encrypt
            
        Returns:
            Encrypted bytes
        """
        return self.fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet.
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            Decrypted bytes
            
        Raises:
            InvalidToken: If decryption fails (wrong key or tampered data)
        """
        try:
            return self.fernet.decrypt(encrypted_data)
        except InvalidToken:
            raise ValueError("Decryption failed: Invalid key or tampered data")
    
    # ==================== HMAC SIGNING ====================
    
    def compute_hmac(self, data: bytes) -> str:
        """Compute HMAC-SHA256 signature for data.
        
        Args:
            data: Data to sign
            
        Returns:
            Hex-encoded HMAC signature
        """
        import hmac
        return hmac.new(self.key, data, hashlib.sha256).hexdigest()
    
    def verify_hmac(self, data: bytes, signature: str) -> bool:
        """Verify HMAC signature.
        
        Args:
            data: Data that was signed
            signature: Expected signature
            
        Returns:
            True if signature matches
        """
        import hmac
        expected = self.compute_hmac(data)
        return hmac.compare_digest(expected, signature)
    
    # ==================== DATABASE HELPERS ====================
    
    def get_encrypted_db_path(self, agent: str, db_type: str) -> Path:
        """Get path to an encrypted database.
        
        Args:
            agent: 'planner' or 'research'
            db_type: 'strategies' or 'apocalypse'
            
        Returns:
            Path to the .enc.db file
        """
        return self.memory_dir / agent / f"{db_type}.enc.db"
    
    def get_runs_db_path(self) -> Path:
        """Get path to the runs database."""
        return self.memory_dir / "runs.db"
    
    def open_encrypted_db(self, agent: str, db_type: str) -> sqlite3.Connection:
        """Open an encrypted SQLite database.
        
        The database file itself is not encrypted, but sensitive columns
        are encrypted at the application level.
        
        Args:
            agent: 'planner' or 'research'
            db_type: 'strategies' or 'apocalypse'
            
        Returns:
            SQLite connection
        """
        db_path = self.get_encrypted_db_path(agent, db_type)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def encrypt_field(self, value: str) -> bytes:
        """Encrypt a string field for database storage.
        
        Args:
            value: String to encrypt
            
        Returns:
            Encrypted bytes suitable for BLOB column
        """
        return self.encrypt(value.encode('utf-8'))
    
    def decrypt_field(self, encrypted_value: bytes) -> str:
        """Decrypt a BLOB field from database.
        
        Args:
            encrypted_value: Encrypted bytes from BLOB column
            
        Returns:
            Decrypted string
        """
        return self.decrypt(encrypted_value).decode('utf-8')


# Singleton instance
_secure_store: Optional[SecureStore] = None

def get_secure_store() -> SecureStore:
    """Get the global SecureStore instance."""
    global _secure_store
    if _secure_store is None:
        _secure_store = SecureStore()
    return _secure_store
