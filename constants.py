from pathlib import Path
from mcp import StdioServerParameters

OLLAMA_MODEL = "llama3.2:latest" # The default Ollama model
# OLLAMA_MODEL = "gemma4:latest" # The default Ollama model

OLLAMA_MODEL_LIST = ["llama3.2:latest", "qwen3:14b", "codegemma:2b", "gemma4:latest", "gemma4:e2b"] # The list of Ollama offline model(s) you have made available

# Ensure first that you used model support thinking. For exmaple Llama3.2 does not:
# "llama3.2:latest" does not support thinking (status code: 400)
THINK_TO_PARSE = None # think: bool | Literal['low', 'medium', 'high'] | None = None,
THINK_TO_ANSWER =  None # think: bool | Literal['low', 'medium', 'high'] | None = None,

DEFAULT_ACTIVE_CONNECTION_NAME = "testpdb01"
ORACLE_SKILLS = "skills-main/db"
MARKDOWN_LANGUAGE = "english"

# CACHE_DIR = Path("cache")
# FILE_SUMMARIES_FILE = CACHE_DIR / "file_summaries.json"
# SNIPPET_STORE_FILE = CACHE_DIR / "snippet_store.json"
# GLOBAL_SUMMARY_FILE = CACHE_DIR / "global_summary.json"
# RETRIEVAL_INDEX_FILE = CACHE_DIR / "retrieval_index.json"
# LOG_FILE = CACHE_DIR / "agent.log"

# Create server parameters for stdio connection
# There are two missing options is my opinion, so far, for SQLcl MCP server:
#   1. The JAVA_TOOL_OPTIONS: -Duser.language=en is not taken into account so feedback might be in local language
#   2. You cannot remove feedback like "xxx rows returned" so it breaks the CSV display with Streamlit
SQLCL_COMMAND = "d:/sqlcl/bin/sql.exe"
SERVER_PARAMS = StdioServerParameters(
  command=SQLCL_COMMAND,
  args=["-mcp"],
)

SKIP_DIRS = {
  ".git", ".venv", "venv", "__pycache__", "node_modules",
  "dist", "build", ".idea", ".vscode"
}

# DEFAULT_MAX_FILE_CHARS = 50000

# KNOWN_DATABASES = {
#   "oracle": "Oracle",
#   "postgresql": "PostgreSQL",
#   "postgres": "PostgreSQL",
#   "mysql": "MySQL",
#   "mariadb": "MariaDB",
#   "sql server": "SQL Server",
#   "mssql": "SQL Server",
#   "sqlite": "SQLite",
#   "mongodb": "MongoDB",
#   "cassandra": "Cassandra",
#   "db2": "DB2",
#   "snowflake": "Snowflake",
#   "redshift": "Redshift",
# }

# KNOWN_TOOLS = {
#   "sqlcl": "SQLcl",
#   "sqlplus": "SQL*Plus",
#   "sql developer": "SQL Developer",
#   "awr": "AWR",
#   "ash": "ASH",
#   "tkprof": "TKPROF",
#   "oem": "OEM",
#   "enterprise manager": "OEM",
#   "rman": "RMAN",
#   "data pump": "Data Pump",
#   "expdp": "Data Pump",
#   "impdp": "Data Pump",
#   "autotrace": "AUTOTRACE",
#   "explain plan": "EXPLAIN PLAN",
#   "dbms_xplan": "DBMS_XPLAN",
#   "dbms_stats": "DBMS_STATS",
#   "aas": "AAS",
# }

# KNOWN_SQL_TOPICS = {
#   "performance": "performance analysis",
#   "tuning": "SQL tuning",
#   "execution plan": "execution plans",
#   "execution plans": "execution plans",
#   "explain plan": "execution plans",
#   "wait event": "wait events",
#   "wait events": "wait events",
#   "active session": "active sessions",
#   "active sessions": "active sessions",
#   "ash": "ASH",
#   "awr": "AWR",
#   "session": "sessions",
#   "sessions": "sessions",
#   "lock": "locking",
#   "locks": "locking",
#   "blocking": "blocking sessions",
#   "index": "indexes",
#   "indexes": "indexes",
#   "partition": "partitioning",
#   "partitions": "partitioning",
#   "join": "joins",
#   "joins": "joins",
#   "parallel": "parallel execution",
#   "histogram": "histograms",
#   "histograms": "histograms",
#   "optimizer": "optimizer",
#   "statistics": "optimizer statistics",
#   "dbms_stats": "optimizer statistics",
#   "plan hash": "plan hash values",
#   "cursor": "cursors",
#   "bind variable": "bind variables",
#   "bind variables": "bind variables",
#   "cte": "common table expressions",
#   "with clause": "common table expressions",
#   "subquery": "subqueries",
#   "temp": "temporary space",
#   "undo": "undo",
#   "redo": "redo",
#   "i/o": "I/O",
#   "io": "I/O",
#   "throughput": "throughput",
#   "latency": "latency",
# }

# STOPWORDS = {
#   "the", "and", "or", "for", "with", "into", "from", "where", "when", "then", "else",
#   "this", "that", "these", "those", "using", "used", "use", "your", "their", "there",
#   "have", "has", "had", "are", "was", "were", "will", "can", "could", "should", "would",
#   "about", "after", "before", "between", "within", "over", "under", "very", "more", "most",
#   "also", "only", "just", "than", "such", "each", "other", "some", "many", "much", "been",
#   "being", "because", "while", "into", "onto", "across", "through", "during", "without",
#   "sql", "select", "from", "where", "group", "order", "join", "left", "right", "inner",
#   "outer", "full", "cross", "on", "as", "by", "asc", "desc", "insert", "update", "delete",
#   "merge", "create", "alter", "drop", "truncate", "begin", "end", "declare", "table",
#   "view", "column", "columns", "rows", "query", "queries", "report", "reports"
# }