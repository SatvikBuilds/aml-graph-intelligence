"""
MongoDB Database Manager for GraphAML Platform
Handles persistent storage for Compliance Officers (Users), Investigation Cases,
Audit Logs, Datasets, and Model Runs with graceful fallback handling.
"""

import os
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

try:
    import pymongo
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

DEFAULT_MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DEFAULT_DB_NAME = "aml_graph_intelligence"

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Generates a PBKDF2-SHA256 password hash with unique salt."""
    if salt is None:
        salt = os.urandom(16).hex()
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
    return key, salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies candidate password against stored PBKDF2 hash."""
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
    return key == stored_hash

class MongoDBManager:
    """Singleton Manager connecting to MongoDB with in-memory resilient fallback."""
    
    def __init__(self, uri: str = DEFAULT_MONGO_URI, db_name: str = DEFAULT_DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.is_connected = False
        self.connection_error = None
        self.last_ping_ms = 0.0

        # In-memory fallback stores (used if MongoDB is unreachable)
        self._memory_users = {}
        self._memory_cases = {}
        self._memory_audit = []
        self._memory_datasets = []
        self._memory_models = []

        self.connect(uri, db_name)

    def connect(self, uri: Optional[str] = None, db_name: Optional[str] = None) -> bool:
        """Establishes connection to MongoDB instance."""
        if uri:
            self.uri = uri
        if db_name:
            self.db_name = db_name

        if not PYMONGO_AVAILABLE:
            self.is_connected = False
            self.connection_error = "pymongo package not installed."
            self._seed_default_users_memory()
            return False

        try:
            t0 = time.time()
            self.client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=2500)
            # Test connection with ping command
            self.client.admin.command("ping")
            self.last_ping_ms = round((time.time() - t0) * 1000, 2)
            self.db = self.client[self.db_name]
            self.is_connected = True
            self.connection_error = None
            
            # Ensure indexes & seed initial users
            self._ensure_indexes()
            self._seed_default_users()
            return True
        except Exception as e:
            self.is_connected = False
            self.connection_error = str(e)
            self._seed_default_users_memory()
            return False

    def _ensure_indexes(self):
        """Creates unique indexes for collections."""
        try:
            if self.is_connected and self.db is not None:
                self.db["users"].create_index("username", unique=True)
                self.db["investigation_cases"].create_index("account_id", unique=True)
                self.db["audit_logs"].create_index("timestamp")
        except Exception:
            pass

    def _seed_default_users(self):
        """Pre-seeds standard compliance officer accounts in MongoDB."""
        defaults = [
            {
                "username": "admin",
                "full_name": "Chief Compliance Officer Sarah Jenkins",
                "role": "Chief AML Officer",
                "password": "admin123"
            },
            {
                "username": "investigator",
                "full_name": "Senior Fraud Analyst Marcus Vance",
                "role": "Senior AML Investigator",
                "password": "aml2025"
            },
            {
                "username": "auditor",
                "full_name": "Risk Compliance Auditor Elena Rostova",
                "role": "Regulatory Compliance Auditor",
                "password": "audit2025"
            }
        ]

        for u in defaults:
            existing = self.db["users"].find_one({"username": u["username"]})
            if not existing:
                p_hash, salt = hash_password(u["password"])
                self.db["users"].insert_one({
                    "username": u["username"],
                    "password_hash": p_hash,
                    "salt": salt,
                    "full_name": u["full_name"],
                    "role": u["role"],
                    "created_at": datetime.utcnow(),
                    "last_login": None
                })

    def _seed_default_users_memory(self):
        """Seeds fallback memory store if MongoDB is offline."""
        defaults = [
            ("admin", "admin123", "Chief Compliance Officer Sarah Jenkins", "Chief AML Officer"),
            ("investigator", "aml2025", "Senior Fraud Analyst Marcus Vance", "Senior AML Investigator"),
            ("auditor", "audit2025", "Risk Compliance Auditor Elena Rostova", "Regulatory Compliance Auditor")
        ]
        for uname, pwd, fname, role in defaults:
            if uname not in self._memory_users:
                p_hash, salt = hash_password(pwd)
                self._memory_users[uname] = {
                    "username": uname,
                    "password_hash": p_hash,
                    "salt": salt,
                    "full_name": fname,
                    "role": role,
                    "created_at": datetime.utcnow(),
                    "last_login": None
                }

    # ==========================================
    # USER AUTHENTICATION & MANAGEMENT
    # ==========================================
    def authenticate_user(self, username: str, candidate_pwd: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Validates credentials against MongoDB or memory store."""
        uname = username.strip().lower()

        if self.is_connected and self.db is not None:
            user = self.db["users"].find_one({"username": uname})
            if not user:
                return False, None, "Invalid username or password."
            if verify_password(candidate_pwd, user["password_hash"], user["salt"]):
                # Update last login
                self.db["users"].update_one(
                    {"username": uname}, 
                    {"$set": {"last_login": datetime.utcnow()}}
                )
                self.log_audit(uname, user["role"], "LOGIN_SUCCESS", "AUTH", "User logged in successfully.")
                safe_user = {
                    "username": user["username"],
                    "full_name": user["full_name"],
                    "role": user["role"],
                    "last_login": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                }
                return True, safe_user, "Authentication successful."
            else:
                self.log_audit(uname, "UNKNOWN", "LOGIN_FAILED", "AUTH", "Incorrect password attempt.")
                return False, None, "Invalid username or password."
        else:
            # Memory fallback
            user = self._memory_users.get(uname)
            if not user:
                return False, None, "Invalid username or password."
            if verify_password(candidate_pwd, user["password_hash"], user["salt"]):
                user["last_login"] = datetime.utcnow()
                safe_user = {
                    "username": user["username"],
                    "full_name": user["full_name"],
                    "role": user["role"],
                    "last_login": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                }
                return True, safe_user, "Authentication successful (Local Memory Fallback)."
            else:
                return False, None, "Invalid username or password."

    def register_user(self, username: str, password: str, full_name: str, role: str) -> Tuple[bool, str]:
        """Registers a new compliance investigator entity."""
        uname = username.strip().lower()
        if len(uname) < 3:
            return False, "Username must contain at least 3 characters."
        if len(password) < 6:
            return False, "Password must contain at least 6 characters."

        p_hash, salt = hash_password(password)
        doc = {
            "username": uname,
            "password_hash": p_hash,
            "salt": salt,
            "full_name": full_name.strip(),
            "role": role,
            "created_at": datetime.utcnow(),
            "last_login": None
        }

        if self.is_connected and self.db is not None:
            try:
                self.db["users"].insert_one(doc)
                self.log_audit("SYSTEM", "ADMIN", "USER_REGISTERED", uname, f"Registered new officer: {full_name} ({role})")
                return True, f"Investigator account '{uname}' registered successfully in MongoDB."
            except pymongo.errors.DuplicateKeyError:
                return False, f"Username '{uname}' is already registered."
            except Exception as e:
                return False, f"Registration failed: {e}"
        else:
            if uname in self._memory_users:
                return False, f"Username '{uname}' already exists."
            self._memory_users[uname] = doc
            return True, f"Investigator account '{uname}' created in local memory."

    # ==========================================
    # INVESTIGATION CASES COLLECTION
    # ==========================================
    def save_case(self, case_dict: Dict[str, Any]) -> bool:
        """Upserts an investigation case file."""
        acc = case_dict.get("account_id")
        if not acc:
            return False

        doc = dict(case_dict)
        doc["updated_at"] = datetime.utcnow()

        if self.is_connected and self.db is not None:
            try:
                self.db["investigation_cases"].update_one(
                    {"account_id": acc},
                    {"$set": doc},
                    upsert=True
                )
                return True
            except Exception:
                return False
        else:
            self._memory_cases[acc] = doc
            return True

    def get_all_cases(self) -> Dict[str, Dict[str, Any]]:
        """Retrieves all tracked cases as a dictionary keyed by account_id."""
        cases = {}
        if self.is_connected and self.db is not None:
            try:
                for c in self.db["investigation_cases"].find({}, {"_id": 0}):
                    cases[c["account_id"]] = c
                return cases
            except Exception:
                return self._memory_cases
        return self._memory_cases

    # ==========================================
    # AUDIT LOGS COLLECTION
    # ==========================================
    def log_audit(self, username: str, role: str, action: str, target: str, details: Optional[str] = None):
        """Records compliance audit events into MongoDB."""
        doc = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "username": username,
            "role": role,
            "action": action,
            "target": target,
            "details": details or ""
        }
        if self.is_connected and self.db is not None:
            try:
                self.db["audit_logs"].insert_one(doc)
            except Exception:
                self._memory_audit.append(doc)
        else:
            self._memory_audit.append(doc)

    def get_recent_audit_logs(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Returns recent audit events."""
        if self.is_connected and self.db is not None:
            try:
                return list(self.db["audit_logs"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
            except Exception:
                return list(reversed(self._memory_audit[-limit:]))
        return list(reversed(self._memory_audit[-limit:]))

    # ==========================================
    # STATUS & METRICS
    # ==========================================
    def get_status(self) -> Dict[str, Any]:
        """Provides real-time MongoDB diagnostics."""
        doc_counts = {}
        if self.is_connected and self.db is not None:
            try:
                for coll_name in ["users", "investigation_cases", "audit_logs", "datasets"]:
                    doc_counts[coll_name] = self.db[coll_name].count_documents({})
            except Exception:
                pass

        return {
            "is_connected": self.is_connected,
            "uri": self.uri,
            "masked_uri": self._mask_uri(self.uri),
            "db_name": self.db_name,
            "ping_ms": self.last_ping_ms,
            "doc_counts": doc_counts,
            "error": self.connection_error,
            "pymongo_installed": PYMONGO_AVAILABLE
        }

    def _mask_uri(self, uri: str) -> str:
        """Hides credentials if URI contains username/password."""
        if "@" in uri:
            prefix = uri.split("@")[0].split("://")[0] + "://"
            host = uri.split("@")[1]
            return f"{prefix}***:***@{host}"
        return uri

# Global singleton instance
_mongo_instance = None

def get_mongo_manager() -> MongoDBManager:
    """Returns singleton MongoDBManager."""
    global _mongo_instance
    if _mongo_instance is None:
        _mongo_instance = MongoDBManager()
    return _mongo_instance
