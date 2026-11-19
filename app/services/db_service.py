"""Database service for Welfog Voice Agent interacting with PostgreSQL."""

import asyncio
import hashlib
import queue
import threading
from datetime import datetime, timedelta
from decimal import Decimal
import math
from typing import Any, Dict, List, Optional, Union
import psycopg2
from psycopg2.extras import RealDictCursor

from app.config import get_settings
from app.utils.logging import logger
from app.utils.pii_masking import mask_sensitive_pii

from app.services.security_service import (
    hash_password,
    verify_password,
    LEGACY_PASSWORD_SALT as PASSWORD_SALT
)


class PooledConnectionWrapper:
    """Wraps a psycopg2 connection so that .close() returns it to the pool instead of closing the TCP socket."""
    def __init__(self, raw_conn, pool: "PostgresConnectionPool"):
        self._raw_conn = raw_conn
        self._pool = pool
        self._is_closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw_conn, name)

    def close(self):
        """Intercept close to return connection to pool."""
        if not self._is_closed:
            self._is_closed = True
            self._pool.return_connection(self._raw_conn)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PostgresConnectionPool:
    """Thread-safe, self-healing PostgreSQL connection pool with health checks and reconnect."""
    def __init__(self, host: str, port: int, user: str, password: str, database: str, max_connections: int = 15):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.max_connections = max_connections
        self._pool: queue.Queue = queue.Queue(maxsize=max_connections)
        self._current_size = 0
        self._lock = threading.Lock()

    def _create_raw_connection(self):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
            cursor_factory=RealDictCursor
        )
        conn.autocommit = True
        return conn

    def get_connection(self, timeout: float = 4.0) -> PooledConnectionWrapper:
        """Get an active pooled connection or create a new one up to max_connections."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                if not conn.closed:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT 1;")
                        return PooledConnectionWrapper(conn, self)
                    except Exception:
                        pass
                try:
                    conn.close()
                except Exception:
                    pass
                with self._lock:
                    self._current_size = max(0, self._current_size - 1)
            except queue.Empty:
                break

        with self._lock:
            if self._current_size < self.max_connections:
                conn = self._create_raw_connection()
                self._current_size += 1
                return PooledConnectionWrapper(conn, self)

        try:
            conn = self._pool.get(timeout=timeout)
            if not conn.closed:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                    return PooledConnectionWrapper(conn, self)
                except Exception:
                    pass
            conn = self._create_raw_connection()
            return PooledConnectionWrapper(conn, self)
        except queue.Empty:
            conn = self._create_raw_connection()
            return PooledConnectionWrapper(conn, self)

    def return_connection(self, conn):
        """Return raw connection to pool or close if excess/closed."""
        try:
            if not conn.closed:
                self._pool.put_nowait(conn)
            else:
                with self._lock:
                    self._current_size = max(0, self._current_size - 1)
        except queue.Full:
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._current_size = max(0, self._current_size - 1)


class DatabaseService:
    """Manages PostgreSQL connections and queries for customers, logs, and authentication."""

    def __init__(self):
        self.settings = get_settings()
        self.host = self.settings.db_host or "localhost"
        self.port = int(self.settings.db_port or 5432)
        self.user = self.settings.db_user or "postgres"
        self.password = self.settings.db_password or "welfog"
        self.database = self.settings.db_name or "welfog_ai"
        self._schema_checked = False
        self._pool: Optional[PostgresConnectionPool] = None
        self._ensure_database_and_tables()
        self._init_pool()

    def _init_pool(self):
        """Initialize connection pool once database is confirmed."""
        try:
            self._pool = PostgresConnectionPool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                max_connections=15
            )
        except Exception as e:
            logger.warning("db_pool_init_warning", error=str(e))

    def _ensure_database_and_tables(self):
        """Self-healing: Creates welfog_ai PostgreSQL database and all required tables if missing."""
        try:
            # 1. Connect to default 'postgres' database to verify / create welfog_ai
            server_conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname="postgres",
                cursor_factory=RealDictCursor
            )
            server_conn.autocommit = True
            with server_conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.database,))
                if not cur.fetchone():
                    cur.execute(f'CREATE DATABASE "{self.database}" ENCODING \'UTF8\';')
            server_conn.close()

            # 2. Connect to welfog_ai to ensure tables, indexes, and initial records
            db_conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.database,
                cursor_factory=RealDictCursor
            )
            db_conn.autocommit = True

            with db_conn.cursor() as cur:
                # Table: admins
                cur.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)

                # Table: customers
                cur.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(150) NOT NULL DEFAULT 'Customer',
                    phone VARCHAR(20) NOT NULL,
                    order_id VARCHAR(50) DEFAULT 'WF-10928',
                    product_name VARCHAR(255) DEFAULT 'Welfog Premium Item',
                    product_price VARCHAR(50) DEFAULT '₹999',
                    delivery_time VARCHAR(100) DEFAULT '2-3 Business Days',
                    payment_method VARCHAR(50) DEFAULT 'Cash on Delivery (COD)',
                    notes TEXT,
                    status VARCHAR(30) DEFAULT 'PENDING',
                    call_duration INT DEFAULT 0,
                    call_summary TEXT,
                    transcript TEXT,
                    recording_url VARCHAR(500) DEFAULT NULL,
                    last_called_at TIMESTAMP DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    qa_score NUMERIC(3,1) DEFAULT NULL,
                    customer_sentiment VARCHAR(50) DEFAULT NULL,
                    rto_risk VARCHAR(50) DEFAULT NULL,
                    qa_notes TEXT DEFAULT NULL,
                    retry_count INT DEFAULT 0,
                    next_retry_at TIMESTAMP DEFAULT NULL,
                    rto_risk_percentage INT DEFAULT NULL,
                    rto_factors TEXT DEFAULT NULL,
                    warehouse_alert SMALLINT DEFAULT 0,
                    recording_local_path VARCHAR(500) DEFAULT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_customers_status ON customers(status);
                CREATE INDEX IF NOT EXISTS idx_customers_phone ON customers(phone);
                CREATE INDEX IF NOT EXISTS idx_customers_order ON customers(order_id);
                """)

                # Table: call_logs
                cur.execute("""
                CREATE TABLE IF NOT EXISTS call_logs (
                    id SERIAL PRIMARY KEY,
                    customer_id INT DEFAULT NULL,
                    phone VARCHAR(20) NOT NULL,
                    customer_name VARCHAR(150) DEFAULT 'Customer',
                    order_id VARCHAR(50) DEFAULT 'WF-10928',
                    status VARCHAR(30) NOT NULL,
                    duration INT DEFAULT 0,
                    summary TEXT,
                    transcript TEXT,
                    recording_url VARCHAR(500) DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    qa_score NUMERIC(3,1) DEFAULT NULL,
                    customer_sentiment VARCHAR(50) DEFAULT NULL,
                    rto_risk VARCHAR(50) DEFAULT NULL,
                    qa_notes TEXT DEFAULT NULL,
                    rto_risk_percentage INT DEFAULT NULL,
                    rto_factors TEXT DEFAULT NULL,
                    warehouse_alert SMALLINT DEFAULT 0,
                    recording_local_path VARCHAR(500) DEFAULT NULL,
                    product_name VARCHAR(255) DEFAULT NULL,
                    product_price VARCHAR(50) DEFAULT NULL,
                    delivery_time VARCHAR(100) DEFAULT NULL,
                    payment_method VARCHAR(50) DEFAULT NULL,
                    notes TEXT DEFAULT NULL,
                    order_outcome VARCHAR(50) DEFAULT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_call_phone ON call_logs(phone);
                CREATE INDEX IF NOT EXISTS idx_call_status ON call_logs(status);
                """)

                # Ensure Default Admin with PBKDF2-HMAC-SHA256 (only on initial setup, never overwrites custom passwords)
                admin_hash = hash_password("admin@123")
                cur.execute("""
                INSERT INTO admins (username, password_hash)
                VALUES ('admin', %s)
                ON CONFLICT (username) DO NOTHING;
                """, (admin_hash,))

                # Dynamically ensure all optional columns in customers
                for col_name, col_def in [
                    ('product_name', 'VARCHAR(255) DEFAULT NULL'),
                    ('product_price', 'VARCHAR(50) DEFAULT NULL'),
                    ('delivery_time', 'VARCHAR(100) DEFAULT NULL'),
                    ('payment_method', 'VARCHAR(50) DEFAULT NULL'),
                    ('qa_score', 'NUMERIC(3,1) DEFAULT NULL'),
                    ('customer_sentiment', 'VARCHAR(50) DEFAULT NULL'),
                    ('rto_risk', 'VARCHAR(50) DEFAULT NULL'),
                    ('qa_notes', 'TEXT DEFAULT NULL'),
                    ('retry_count', 'INT DEFAULT 0'),
                    ('next_retry_at', 'TIMESTAMP DEFAULT NULL'),
                    ('rto_risk_percentage', 'INT DEFAULT NULL'),
                    ('rto_factors', 'TEXT DEFAULT NULL'),
                    ('warehouse_alert', 'SMALLINT DEFAULT 0'),
                    ('recording_local_path', 'VARCHAR(500) DEFAULT NULL')
                ]:
                    cur.execute(f"ALTER TABLE customers ADD COLUMN IF NOT EXISTS {col_name} {col_def};")

                # Dynamically ensure all optional columns in call_logs
                for col_name, col_def in [
                    ('qa_score', 'NUMERIC(3,1) DEFAULT NULL'),
                    ('customer_sentiment', 'VARCHAR(50) DEFAULT NULL'),
                    ('rto_risk', 'VARCHAR(50) DEFAULT NULL'),
                    ('qa_notes', 'TEXT DEFAULT NULL'),
                    ('rto_risk_percentage', 'INT DEFAULT NULL'),
                    ('rto_factors', 'TEXT DEFAULT NULL'),
                    ('warehouse_alert', 'SMALLINT DEFAULT 0'),
                    ('recording_local_path', 'VARCHAR(500) DEFAULT NULL'),
                    ('product_name', 'VARCHAR(255) DEFAULT NULL'),
                    ('product_price', 'VARCHAR(50) DEFAULT NULL'),
                    ('delivery_time', 'VARCHAR(100) DEFAULT NULL'),
                    ('payment_method', 'VARCHAR(50) DEFAULT NULL'),
                    ('notes', 'TEXT DEFAULT NULL'),
                    ('order_outcome', 'VARCHAR(50) DEFAULT NULL')
                ]:
                    cur.execute(f"ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS {col_name} {col_def};")

                # Seed initial sample records if customer table is empty
                cur.execute("SELECT COUNT(*) AS cnt FROM customers")
                if cur.fetchone()["cnt"] == 0:
                    seed_data = [
                        ('Sidharth Sharma', '9828628181', 'WF-10928', 'Noise-Cancelling Wireless Headphones', '₹1,499', 'Tomorrow by 4 PM', 'Cash on Delivery (COD)', 'Plot 42, Near Metro Station, Mansarovar, Jaipur (302020)', 'PENDING'),
                        ('Abhishek Degra', '9521399708', 'WF-10929', 'Smart Fitness Watch Series 5', '₹2,299', 'In 2 Days', 'Prepaid (UPI)', 'House No 12, Malviya Nagar, Sector 4, Jaipur (302017)', 'PENDING'),
                        ('Lokesh Choudhary', '9358991311', 'WF-10930', 'Fast Charging Power Bank 20000mAh', '₹999', 'Tomorrow by 1 PM', 'Cash on Delivery (COD)', 'Vijay Lal Degra ki Dhani, Sinwar, Jaipur', 'PENDING')
                    ]
                    cur.executemany("""
                    INSERT INTO customers (name, phone, order_id, product_name, product_price, delivery_time, payment_method, notes, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, seed_data)

            db_conn.close()
            self._schema_checked = True
            logger.info("db_schema_ready", database=self.database)
        except Exception as e:
            logger.error("db_ensure_database_and_tables_error", error=str(e))

    def _get_connection(self):
        """Acquire a pooled connection with auto-healing resilience."""
        if not self._pool:
            self._init_pool()

        try:
            if self._pool:
                return self._pool.get_connection()
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.database,
                cursor_factory=RealDictCursor
            )
            conn.autocommit = True
            return conn
        except (psycopg2.OperationalError, psycopg2.ProgrammingError) as e:
            logger.info("db_auto_healing_triggered", reason=str(e))
            self._ensure_database_and_tables()
            self._init_pool()
            if self._pool:
                return self._pool.get_connection()
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.database,
                cursor_factory=RealDictCursor
            )
            conn.autocommit = True
            return conn

    async def verify_admin(self, username: str, password: str) -> bool:
        """Verify admin login credentials with PBKDF2-HMAC-SHA256 and transparent legacy rehash."""
        def _verify():
            try:
                conn = self._get_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT password_hash FROM admins WHERE username = %s LIMIT 1",
                        (username,)
                    )
                    row = cur.fetchone()
                    if not row or not row.get("password_hash"):
                        return False

                    stored_hash = row["password_hash"]
                    is_valid, needs_rehash = verify_password(password, stored_hash)

                    if is_valid and needs_rehash:
                        # Auto-upgrade legacy hash to PBKDF2-HMAC-SHA256 (600,000 rounds)
                        try:
                            upgraded_hash = hash_password(password)
                            cur.execute(
                                "UPDATE admins SET password_hash = %s WHERE username = %s",
                                (upgraded_hash, username)
                            )
                            logger.info("admin_password_hash_upgraded_to_pbkdf2", username=username)
                        except Exception as rehash_err:
                            logger.warning("failed_upgrading_password_hash", error=str(rehash_err))

                    return is_valid
            except Exception as e:
                logger.error("db_verify_admin_error", error=str(e))
                return False
            finally:
                if 'conn' in locals() and conn:
                    conn.close()

        return await asyncio.to_thread(_verify)

    async def update_admin_password(self, username: str, new_password: str) -> bool:
        """Update admin password with PBKDF2-HMAC-SHA256 hash."""
        def _update():
            new_hash = hash_password(new_password)
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE admins SET password_hash = %s WHERE username = %s",
                        (new_hash, username)
                    )
                    return cur.rowcount > 0
            finally:
                conn.close()

        return await asyncio.to_thread(_update)

    async def get_customers(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all onboarded customers with optional status or text filter."""
        def _fetch():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    query = "SELECT * FROM customers WHERE 1=1"
                    params = []

                    if status and status.upper() != "ALL":
                        query += " AND status = %s"
                        params.append(status.upper())

                    if search:
                        query += " AND (name ILIKE %s OR phone ILIKE %s OR order_id ILIKE %s)"
                        like_val = f"%{search}%"
                        params.extend([like_val, like_val, like_val])

                    query += """ ORDER BY 
                        CASE UPPER(status)
                            WHEN 'CALLING' THEN 1
                            WHEN 'CONNECTED' THEN 2
                            WHEN 'PENDING' THEN 3
                            WHEN 'COMPLETED' THEN 4
                            WHEN 'NO_ANSWER' THEN 5
                            WHEN 'BUSY' THEN 6
                            WHEN 'FAILED' THEN 7
                            ELSE 8
                        END, id ASC"""

                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()
                    result = []
                    for r in rows:
                        item = dict(r)
                        for k in ["created_at", "updated_at", "last_called_at", "next_retry_at"]:
                            if isinstance(item.get(k), datetime):
                                item[k] = item[k].strftime("%Y-%m-%d %H:%M:%S")
                        result.append(item)
                    return result
            finally:
                conn.close()

        return await asyncio.to_thread(_fetch)

    async def get_customer(self, customer_id: int) -> Optional[Dict[str, Any]]:
        """Fetch customer by ID."""
        def _fetch():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM customers WHERE id = %s", (customer_id,))
                    row = cur.fetchone()
                    if row:
                        item = dict(row)
                        for k in ["created_at", "updated_at", "last_called_at", "next_retry_at"]:
                            if isinstance(item.get(k), datetime):
                                item[k] = item[k].strftime("%Y-%m-%d %H:%M:%S")
                        return item
                    return None
            finally:
                conn.close()

        return await asyncio.to_thread(_fetch)

    async def get_next_pending_customer(self) -> Optional[Dict[str, Any]]:
        """Retrieve the next pending customer in the calling queue."""
        def _fetch():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM customers WHERE status = 'PENDING' ORDER BY id ASC LIMIT 1"
                    )
                    row = cur.fetchone()
                    if row:
                        item = dict(row)
                        for k in ["created_at", "updated_at", "last_called_at", "next_retry_at"]:
                            if isinstance(item.get(k), datetime):
                                item[k] = item[k].strftime("%Y-%m-%d %H:%M:%S")
                        return item
                    return None
            finally:
                conn.close()

        return await asyncio.to_thread(_fetch)

    async def add_customer(
        self,
        name: str,
        phone: str,
        order_id: Optional[str] = None,
        notes: Optional[str] = None,
        product_name: Optional[str] = None,
        product_price: Optional[str] = None,
        delivery_time: Optional[str] = None,
        payment_method: Optional[str] = None
    ) -> int:
        """Insert a single customer into queue with rich order context."""
        def _insert():
            cleaned_phone = "".join(c for c in phone if c.isdigit())
            if len(cleaned_phone) == 12 and cleaned_phone.startswith("91"):
                cleaned_phone = cleaned_phone[2:]

            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO customers (
                            name, phone, order_id, notes, status,
                            product_name, product_price, delivery_time, payment_method
                        )
                        VALUES (%s, %s, %s, %s, 'PENDING', %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (
                            name.strip() or "Customer",
                            cleaned_phone,
                            (order_id or "WF-10928").strip(),
                            notes or "",
                            product_name.strip() if product_name else "Welfog Premium Item",
                            product_price.strip() if product_price else "₹999",
                            delivery_time.strip() if delivery_time else "2-3 Business Days",
                            payment_method.strip() if payment_method else "Cash on Delivery (COD)"
                        )
                    )
                    row = cur.fetchone()
                    return row["id"]
            finally:
                conn.close()

        return await asyncio.to_thread(_insert)

    async def bulk_add_customers(self, entries: List[Dict[str, str]]) -> int:
        """Bulk insert customers from list of dicts with optional order metadata."""
        def _bulk_insert():
            conn = self._get_connection()
            try:
                count = 0
                with conn.cursor() as cur:
                    for item in entries:
                        raw_phone = item.get("phone", "")
                        cleaned = "".join(c for c in raw_phone if c.isdigit())
                        if len(cleaned) == 12 and cleaned.startswith("91"):
                            cleaned = cleaned[2:]
                        if not cleaned or len(cleaned) < 10:
                            continue

                        cur.execute(
                            """
                            INSERT INTO customers (
                                name, phone, order_id, notes, status,
                                product_name, product_price, delivery_time, payment_method
                            )
                            VALUES (%s, %s, %s, %s, 'PENDING', %s, %s, %s, %s);
                            """,
                            (
                                item.get("name", "Customer").strip() or "Customer",
                                cleaned,
                                item.get("order_id", "WF-10928").strip() or "WF-10928",
                                item.get("notes", "Bulk Onboarded"),
                                item.get("product_name") or "Welfog Premium Item",
                                item.get("product_price") or "₹999",
                                item.get("delivery_time") or "2-3 Business Days",
                                item.get("payment_method") or "Cash on Delivery (COD)"
                            )
                        )
                        count += 1
                return count
            finally:
                conn.close()

        return await asyncio.to_thread(_bulk_insert)

    async def update_customer(
        self,
        customer_id: int,
        name: str,
        phone: str,
        order_id: Optional[str] = None,
        notes: Optional[str] = None,
        product_name: Optional[str] = None,
        product_price: Optional[str] = None,
        delivery_time: Optional[str] = None,
        payment_method: Optional[str] = None
    ) -> bool:
        """Edit customer info and order metadata."""
        def _edit():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cleaned = "".join(c for c in phone if c.isdigit())
                    if len(cleaned) == 12 and cleaned.startswith("91"):
                        cleaned = cleaned[2:]
                    cur.execute(
                        """
                        UPDATE customers 
                        SET name = %s, phone = %s, order_id = %s, notes = %s,
                            product_name = COALESCE(%s, product_name),
                            product_price = COALESCE(%s, product_price),
                            delivery_time = COALESCE(%s, delivery_time),
                            payment_method = COALESCE(%s, payment_method),
                            updated_at = NOW()
                        WHERE id = %s;
                        """,
                        (
                            name.strip(),
                            cleaned,
                            order_id.strip() if order_id else "WF-10928",
                            notes or "",
                            product_name,
                            product_price,
                            delivery_time,
                            payment_method,
                            customer_id
                        )
                    )
                    return True
            finally:
                conn.close()

        return await asyncio.to_thread(_edit)

    async def update_customer_status(
        self,
        customer_id: int,
        status: str,
        call_duration: Optional[int] = None,
        call_summary: Optional[str] = None,
        transcript: Optional[str] = None,
        recording_url: Optional[str] = None,
        qa_score: Optional[float] = None,
        customer_sentiment: Optional[str] = None,
        rto_risk: Optional[str] = None,
        qa_notes: Optional[str] = None,
        rto_risk_percentage: Optional[int] = None,
        rto_factors: Optional[str] = None,
        warehouse_alert: Optional[bool] = None,
        recording_local_path: Optional[str] = None
    ) -> bool:
        """Update customer call status and record notes/transcript/recording/QA scores and warehouse flags."""
        def _update():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    updates = ["status = %s", "last_called_at = NOW()", "updated_at = NOW()"]
                    params = [status.upper()]

                    if call_duration is not None:
                        updates.append("call_duration = %s")
                        params.append(call_duration)

                    if call_summary is not None:
                        updates.append("call_summary = %s")
                        params.append(mask_sensitive_pii(call_summary))

                    if transcript is not None:
                        updates.append("transcript = %s")
                        params.append(mask_sensitive_pii(transcript))

                    if recording_url is not None:
                        updates.append("recording_url = %s")
                        params.append(recording_url)

                    if qa_score is not None:
                        updates.append("qa_score = %s")
                        params.append(qa_score)

                    if customer_sentiment is not None:
                        updates.append("customer_sentiment = %s")
                        params.append(customer_sentiment.upper())

                    if rto_risk is not None:
                        updates.append("rto_risk = %s")
                        params.append(rto_risk.upper())

                    if qa_notes is not None:
                        updates.append("qa_notes = %s")
                        params.append(mask_sensitive_pii(qa_notes))

                    if rto_risk_percentage is not None:
                        updates.append("rto_risk_percentage = %s")
                        params.append(rto_risk_percentage)

                    if rto_factors is not None:
                        updates.append("rto_factors = %s")
                        params.append(mask_sensitive_pii(rto_factors))

                    if warehouse_alert is not None:
                        updates.append("warehouse_alert = %s")
                        params.append(1 if warehouse_alert else 0)

                    if recording_local_path is not None:
                        updates.append("recording_local_path = %s")
                        params.append(recording_local_path)

                    params.append(customer_id)
                    cur.execute(
                        f"UPDATE customers SET {', '.join(updates)} WHERE id = %s;",
                        tuple(params)
                    )
                    return True
            finally:
                conn.close()

        return await asyncio.to_thread(_update)

    async def schedule_retry(self, customer_id: int, delay_minutes: int = 15) -> bool:
        """Schedule automatic retry for failed / busy / no-answer customer."""
        def _schedule():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    next_time = datetime.now() + timedelta(minutes=delay_minutes)
                    cur.execute(
                        """
                        UPDATE customers 
                        SET retry_count = retry_count + 1,
                            next_retry_at = %s,
                            status = 'SCHEDULED_RETRY',
                            updated_at = NOW()
                        WHERE id = %s AND retry_count < 3;
                        """,
                        (next_time, customer_id)
                    )
                    return cur.rowcount > 0
            finally:
                conn.close()

        return await asyncio.to_thread(_schedule)

    async def get_scheduled_retries(self) -> List[Dict[str, Any]]:
        """Fetch all customers whose retry timer is due and attempts < 3."""
        def _fetch():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM customers 
                        WHERE (status = 'SCHEDULED_RETRY' OR status IN ('NO_ANSWER', 'BUSY', 'FAILED'))
                          AND retry_count < 3
                          AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                        ORDER BY next_retry_at ASC;
                        """
                    )
                    rows = cur.fetchall()
                    result = []
                    for r in rows:
                        item = dict(r)
                        for k in ["created_at", "updated_at", "last_called_at", "next_retry_at"]:
                            if isinstance(item.get(k), datetime):
                                item[k] = item[k].strftime("%Y-%m-%d %H:%M:%S")
                        result.append(item)
                    return result
            finally:
                conn.close()

        return await asyncio.to_thread(_fetch)

    async def get_high_rto_customers(self) -> List[Dict[str, Any]]:
        """Fetch customers flagged with high RTO risk for warehouse team."""
        def _fetch():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM customers 
                        WHERE warehouse_alert = 1 OR rto_risk = 'HIGH' OR rto_risk_percentage >= 70
                        ORDER BY id DESC;
                        """
                    )
                    rows = cur.fetchall()
                    result = []
                    for r in rows:
                        item = dict(r)
                        for k in ["created_at", "updated_at", "last_called_at"]:
                            if isinstance(item.get(k), datetime):
                                item[k] = item[k].strftime("%Y-%m-%d %H:%M:%S")
                        result.append(item)
                    return result
            finally:
                conn.close()

        return await asyncio.to_thread(_fetch)

    async def delete_customer(self, customer_id: int) -> bool:
        """Delete customer by ID."""
        def _delete():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM customers WHERE id = %s;", (customer_id,))
                    return True
            finally:
                conn.close()

        return await asyncio.to_thread(_delete)

    async def reset_customer(self, customer_id: int) -> bool:
        """Reset customer back to PENDING."""
        def _reset():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE customers SET status = 'PENDING', call_duration = 0, call_summary = NULL, updated_at = NOW() WHERE id = %s;",
                        (customer_id,)
                    )
                    return True
            finally:
                conn.close()

        return await asyncio.to_thread(_reset)

    async def reset_all_customers(self) -> int:
        """Reset all completed or failed customers back to PENDING for a new campaign."""
        def _reset_all():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE customers SET status = 'PENDING', call_duration = 0, call_summary = NULL, updated_at = NOW() WHERE status != 'PENDING';"
                    )
                    return cur.rowcount
            finally:
                conn.close()

        return await asyncio.to_thread(_reset_all)

    async def reset_unanswered_customers(self) -> int:
        """Reset only NO_ANSWER, BUSY, or FAILED customers back to PENDING for re-calling."""
        def _reset_unanswered():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE customers SET status = 'PENDING', call_duration = 0, call_summary = NULL, updated_at = NOW() WHERE status IN ('NO_ANSWER', 'BUSY', 'FAILED');"
                    )
                    return cur.rowcount
            finally:
                conn.close()

        return await asyncio.to_thread(_reset_unanswered)

    async def record_call_log(
        self,
        customer_id: Optional[int],
        phone: str,
        customer_name: str,
        order_id: str,
        status: str,
        duration: int,
        summary: Optional[str] = None,
        transcript: Optional[str] = None,
        recording_url: Optional[str] = None,
        qa_score: Optional[float] = None,
        customer_sentiment: Optional[str] = None,
        rto_risk: Optional[str] = None,
        qa_notes: Optional[str] = None,
        product_name: Optional[str] = None,
        product_price: Optional[str] = None,
        delivery_time: Optional[str] = None,
        payment_method: Optional[str] = None,
        notes: Optional[str] = None,
        order_outcome: Optional[str] = None,
        rto_risk_percentage: Optional[int] = None,
        rto_factors: Optional[str] = None,
        warehouse_alert: Optional[Union[bool, int]] = None
    ) -> int:
        """Insert historical call record into call_logs table with QA metrics and real customer order context."""
        def _insert():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    p_name = product_name
                    p_price = product_price
                    p_deliv = delivery_time
                    p_pay = payment_method
                    p_notes = notes
                    p_outcome = order_outcome or status.upper()

                    if customer_id and (not p_name or not p_price or not p_notes):
                        try:
                            cur.execute(
                                "SELECT product_name, product_price, delivery_time, payment_method, notes, status FROM customers WHERE id = %s",
                                (customer_id,)
                            )
                            c_row = cur.fetchone()
                            if c_row:
                                p_name = p_name or c_row.get("product_name")
                                p_price = p_price or c_row.get("product_price")
                                p_deliv = p_deliv or c_row.get("delivery_time")
                                p_pay = p_pay or c_row.get("payment_method")
                                p_notes = p_notes or c_row.get("notes")
                                if not order_outcome:
                                    p_outcome = c_row.get("status") or status.upper()
                        except Exception as e:
                            logger.warning("record_call_log_customer_lookup_warn", error=str(e))

                    # Dynamic AI-driven Outcome & RTO Calculation
                    calc_outcome = (p_outcome or status or "PENDING").upper()
                    is_cancelled = (calc_outcome == "CANCELLED" or status.upper() == "CANCELLED")

                    calc_rto_risk = (
                        rto_risk.upper() if rto_risk
                        else ("HIGH" if is_cancelled else ("MEDIUM" if status.upper() in {"NO_ANSWER", "BUSY"} else "LOW"))
                    )
                    calc_rto_pct = (
                        rto_risk_percentage if rto_risk_percentage is not None
                        else (95 if is_cancelled else (50 if status.upper() in {"NO_ANSWER", "BUSY"} else 15))
                    )
                    calc_wh_alert = 1 if (is_cancelled or calc_rto_risk == "HIGH" or (warehouse_alert is not None and warehouse_alert)) else 0
                    calc_rto_factors = rto_factors or (
                        "Order marked cancelled - Do Not Dispatch" if is_cancelled
                        else ("Customer did not answer call" if status.upper() in {"NO_ANSWER", "BUSY"} else "Verified by AI")
                    )
                    calc_sentiment = (
                        customer_sentiment or ("FRUSTRATED" if is_cancelled else "NEUTRAL")
                    ).upper()

                    cur.execute(
                        """
                        INSERT INTO call_logs (
                            customer_id, phone, customer_name, order_id, status, duration, 
                            summary, transcript, recording_url, qa_score, customer_sentiment, rto_risk, qa_notes,
                            rto_risk_percentage, rto_factors, warehouse_alert,
                            product_name, product_price, delivery_time, payment_method, notes, order_outcome
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (
                            customer_id, phone, customer_name, order_id, status.upper(), duration,
                            mask_sensitive_pii(summary or ""),
                            mask_sensitive_pii(transcript or ""),
                            recording_url or "",
                            qa_score if qa_score is not None else 9.0,
                            calc_sentiment,
                            calc_rto_risk,
                            mask_sensitive_pii(qa_notes or ""),
                            calc_rto_pct,
                            calc_rto_factors,
                            calc_wh_alert,
                            p_name or "",
                            p_price or "",
                            p_deliv or "",
                            p_pay or "",
                            mask_sensitive_pii(p_notes or ""),
                            calc_outcome
                        )
                    )
                    row = cur.fetchone()
                    return row["id"]
            finally:
                conn.close()

        return await asyncio.to_thread(_insert)

    async def update_call_log(
        self,
        log_id: int,
        qa_score: Optional[float] = None,
        customer_sentiment: Optional[str] = None,
        rto_risk: Optional[str] = None,
        rto_risk_percentage: Optional[int] = None,
        rto_factors: Optional[str] = None,
        warehouse_alert: Optional[Union[bool, int]] = None,
        qa_notes: Optional[str] = None,
        order_outcome: Optional[str] = None,
        summary: Optional[str] = None,
        transcript: Optional[str] = None,
        status: Optional[str] = None
    ) -> bool:
        """Update historical call log record with audited QA score, sentiment, RTO risk, and full transcript."""
        def _update():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    updates = []
                    params = []

                    if qa_score is not None:
                        updates.append("qa_score = %s")
                        params.append(qa_score)
                    if customer_sentiment is not None:
                        updates.append("customer_sentiment = %s")
                        params.append(customer_sentiment.upper())
                    if rto_risk is not None:
                        updates.append("rto_risk = %s")
                        params.append(rto_risk.upper())
                    if rto_risk_percentage is not None:
                        updates.append("rto_risk_percentage = %s")
                        params.append(rto_risk_percentage)
                    if rto_factors is not None:
                        updates.append("rto_factors = %s")
                        params.append(rto_factors)
                    if warehouse_alert is not None:
                        updates.append("warehouse_alert = %s")
                        params.append(1 if warehouse_alert else 0)
                    if qa_notes is not None:
                        updates.append("qa_notes = %s")
                        params.append(mask_sensitive_pii(qa_notes))
                    if order_outcome is not None:
                        updates.append("order_outcome = %s")
                        params.append(order_outcome.upper())
                    if status is not None:
                        updates.append("status = %s")
                        params.append(status.upper())
                    if summary is not None:
                        updates.append("summary = %s")
                        params.append(mask_sensitive_pii(summary))
                    if transcript is not None:
                        updates.append("transcript = %s")
                        params.append(mask_sensitive_pii(transcript))

                    if not updates:
                        return False

                    params.append(log_id)
                    cur.execute(f"UPDATE call_logs SET {', '.join(updates)} WHERE id = %s;", tuple(params))
                    return True
            finally:
                conn.close()

        return await asyncio.to_thread(_update)

    async def get_call_logs(
        self,
        limit: Optional[int] = None,
        page: int = 1,
        page_size: int = 10,
        status: Optional[str] = None,
        search: Optional[str] = None,
        return_pagination: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Fetch recent call logs with pagination, search, status filtering and real customer data."""
        def _fetch():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    where_clauses = []
                    params = []

                    if status and status.upper() not in {"ALL", ""}:
                        st = status.upper().strip()
                        if st == "COMPLETED":
                            where_clauses.append("UPPER(l.status) = 'COMPLETED'")
                        elif st == "CONFIRMED":
                            where_clauses.append("(UPPER(COALESCE(l.order_outcome, '')) = 'CONFIRMED' OR UPPER(COALESCE(c.status, '')) = 'CONFIRMED' OR UPPER(l.summary) LIKE %s OR UPPER(l.transcript) LIKE %s)")
                            params.extend(["%CONFIRM%", "%CONFIRM%"])
                        elif st == "CANCELLED":
                            where_clauses.append("(UPPER(COALESCE(l.order_outcome, '')) = 'CANCELLED' OR UPPER(COALESCE(c.status, '')) = 'CANCELLED' OR UPPER(l.summary) LIKE %s OR UPPER(l.transcript) LIKE %s)")
                            params.extend(["%CANCEL%", "%CANCEL%"])
                        elif st in {"NO_ANSWER", "BUSY"}:
                            where_clauses.append("UPPER(l.status) IN ('NO_ANSWER', 'BUSY')")
                        else:
                            where_clauses.append("UPPER(l.status) = %s")
                            params.append(st)

                    if search and search.strip():
                        s = f"%{search.strip()}%"
                        where_clauses.append(
                            "(l.customer_name ILIKE %s OR l.phone ILIKE %s OR l.order_id ILIKE %s OR "
                            "COALESCE(l.product_name, c.product_name, '') ILIKE %s OR "
                            "COALESCE(l.notes, c.notes, '') ILIKE %s OR "
                            "COALESCE(l.summary, '') ILIKE %s)"
                        )
                        params.extend([s, s, s, s, s, s])

                    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

                    # Count total records
                    count_query = f"""
                        SELECT COUNT(*) AS total
                        FROM call_logs l
                        LEFT JOIN customers c ON (l.customer_id = c.id OR (l.customer_id IS NULL AND l.phone = c.phone))
                        {where_sql};
                    """
                    cur.execute(count_query, tuple(params))
                    total_count = cur.fetchone()["total"]

                    # Fetch rows with real customer & order info
                    select_query = f"""
                        SELECT 
                            l.id,
                            l.customer_id,
                            COALESCE(NULLIF(l.customer_name, ''), c.name, 'Customer') AS customer_name,
                            l.phone,
                            COALESCE(NULLIF(l.order_id, ''), c.order_id, 'N/A') AS order_id,
                            l.status,
                            l.duration,
                            l.summary,
                            l.transcript,
                            l.recording_url,
                            l.created_at,
                            l.qa_score,
                            l.customer_sentiment,
                            l.rto_risk,
                            l.qa_notes,
                            l.rto_risk_percentage,
                            l.rto_factors,
                            l.warehouse_alert,
                            l.recording_local_path,
                            COALESCE(NULLIF(l.product_name, ''), c.product_name, 'Welfog Order Item') AS product_name,
                            COALESCE(NULLIF(l.product_price, ''), c.product_price, '') AS product_price,
                            COALESCE(NULLIF(l.delivery_time, ''), c.delivery_time, '2-3 Business Days') AS delivery_time,
                            COALESCE(NULLIF(l.payment_method, ''), c.payment_method, 'Cash on Delivery (COD)') AS payment_method,
                            COALESCE(NULLIF(l.notes, ''), c.notes, '') AS customer_notes,
                            COALESCE(NULLIF(l.order_outcome, ''), c.status, l.status) AS order_outcome
                        FROM call_logs l
                        LEFT JOIN customers c ON (l.customer_id = c.id OR (l.customer_id IS NULL AND l.phone = c.phone))
                        {where_sql}
                        ORDER BY l.id DESC
                    """

                    query_params = list(params)
                    if limit is not None and not return_pagination:
                        select_query += " LIMIT %s;"
                        query_params.append(limit)
                    else:
                        safe_page = max(1, page)
                        safe_page_size = max(1, page_size)
                        offset = (safe_page - 1) * safe_page_size
                        select_query += " LIMIT %s OFFSET %s;"
                        query_params.extend([safe_page_size, offset])

                    cur.execute(select_query, tuple(query_params))
                    rows = cur.fetchall()
                    result_rows = []
                    for r in rows:
                        item = dict(r)
                        if isinstance(item.get("created_at"), datetime):
                            item["created_at"] = item["created_at"].strftime("%Y-%m-%d %H:%M:%S")
                        if isinstance(item.get("qa_score"), Decimal):
                            item["qa_score"] = float(item["qa_score"])
                        result_rows.append(item)

                    if return_pagination:
                        total_pages = max(1, math.ceil(total_count / safe_page_size)) if total_count > 0 else 1
                        return {
                            "logs": result_rows,
                            "count": len(result_rows),
                            "total": total_count,
                            "page": safe_page,
                            "page_size": safe_page_size,
                            "total_pages": total_pages,
                            "has_prev": safe_page > 1,
                            "has_next": safe_page < total_pages
                        }
                    return result_rows
            finally:
                conn.close()

        return await asyncio.to_thread(_fetch)

    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """Aggregate high-level metrics for dashboard cards."""
        def _stats():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) AS total FROM customers;")
                    total = cur.fetchone()["total"]

                    cur.execute("SELECT COUNT(*) AS pending FROM customers WHERE status = 'PENDING';")
                    pending = cur.fetchone()["pending"]

                    cur.execute("SELECT COUNT(*) AS completed FROM customers WHERE status = 'COMPLETED';")
                    completed = cur.fetchone()["completed"]

                    cur.execute("SELECT COUNT(*) AS no_answer FROM customers WHERE status = 'NO_ANSWER';")
                    no_answer = cur.fetchone()["no_answer"]

                    cur.execute("SELECT COUNT(*) AS failed FROM customers WHERE status IN ('FAILED', 'BUSY');")
                    failed = cur.fetchone()["failed"]

                    cur.execute("SELECT COALESCE(SUM(duration), 0) AS total_duration FROM call_logs;")
                    total_duration = cur.fetchone()["total_duration"]

                    cur.execute("""
                        SELECT COUNT(*) AS high_rto 
                        FROM customers 
                        WHERE warehouse_alert = 1 OR rto_risk = 'HIGH' OR rto_risk_percentage >= 70;
                    """)
                    high_rto = cur.fetchone()["high_rto"]

                    cur.execute("SELECT COUNT(*) AS attempted FROM customers WHERE status NOT IN ('PENDING', 'CALLING');")
                    attempted = cur.fetchone()["attempted"]
                    rate = round((completed / attempted * 100), 1) if attempted > 0 else 0.0

                    return {
                        "total_customers": total,
                        "pending_calls": pending,
                        "completed_calls": completed,
                        "no_answer_calls": no_answer,
                        "failed_calls": failed,
                        "total_duration_seconds": int(total_duration),
                        "success_rate_percent": rate,
                        "high_rto_alerts": high_rto
                    }
            finally:
                conn.close()

        return await asyncio.to_thread(_stats)


_db_service: Optional[DatabaseService] = None


def get_db_service() -> DatabaseService:
    """Return singleton DatabaseService instance."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service
