class Config:
    env_file = ".env"

MODEL_PATH = "./models/trained"
API_TIMEOUT = 30
MAX_CONCURRENT_REQUESTS = 100
LOG_LEVEL = "INFO"
DATABASE_URL = "postgresql://kaagapai:kaagapai_password@localhost:5432/kaagapai"
REDIS_URL = "redis://localhost:6379"
JWT_SECRET_KEY = "your-super-secret-jwt-key-here"
API_KEY = "demo-token"
MONTE_CARLO_SIMULATIONS = 1000
SHAP_SAMPLE_SIZE = 100
ENABLE_BIAS_MONITORING = True
DEFAULT_LANGUAGE = "english"
ENABLE_CULTURAL_EXPLANATIONS = True
ENABLE_PROMETHEUS_METRICS = True
METRICS_PORT = 9090
