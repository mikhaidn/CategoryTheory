# Phase 7: MLOps & Production ML - Exercises & Self-Assessment

> Apply your SWE skills to build production ML systems

## Your Advantage

As a professional SWE with architecture experience, this phase is where you'll excel. Most ML practitioners struggle here - you won't.

---

## 7.1 ML System Design - Exercises

### Design Challenges 🏗️

**DC1.1**: Design a recommendation system
- Requirements: 10M users, real-time recommendations, personalized
- Design: Data pipeline, model training, serving, monitoring
- Consider: Latency, scalability, cold start, freshness
- Draw: Architecture diagram

**DC1.2**: Design a fraud detection system
- Requirements: Process 1000 transactions/sec, <100ms latency, high recall
- Design: Feature engineering, model, real-time inference
- Consider: Class imbalance, concept drift, explainability
- Draw: End-to-end system

**DC1.3**: Design an LLM-powered chatbot
- Requirements: Customer support, context-aware, accurate
- Design: RAG vs fine-tuning, conversation memory, moderation
- Consider: Cost, latency, safety, evaluation
- Draw: System architecture

### Concept Checks ✓

**CC1.1**: Batch vs online serving - when to use each?

**CC1.2**: How do you handle model versioning in production?

**CC1.3**: Explain technical debt in ML systems. Give 5 examples.

**CC1.4**: How would you A/B test a new model in production?

### Checkpoint 🚦
- [ ] Can design end-to-end ML systems?
- [ ] Understand tradeoffs (latency, cost, accuracy)?
- [ ] Think about production concerns upfront?

---

## 7.2 Experiment Tracking - Exercises

### Must-Implement ✓

**Experiment Tracking System**
- [ ] Set up MLflow or Weights & Biases
- [ ] Track hyperparameters, metrics, artifacts
- [ ] Compare experiments
- [ ] Version datasets with DVC

### Coding Exercises 💻

**CE2.1**: MLflow integration
```python
import mlflow

class MLExperiment:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def run_experiment(self, model, params, train_data, val_data):
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Train model
            model.fit(train_data)

            # Evaluate
            metrics = self.evaluate(model, val_data)
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Log artifacts (plots, etc.)
            self.log_artifacts()

    # YOUR CODE HERE
```

**CE2.2**: Data versioning with DVC
```bash
# Set up DVC
# Version a dataset
# Track changes
# Integrate with experiment tracking
```

### Applied Problems 🎯

**AP2.1**: Reproducible ML pipeline
- Pick a previous project
- Add MLflow tracking
- Version data with DVC
- Make fully reproducible
- Document setup process

### Checkpoint 🚦
- [ ] Using experiment tracking in all projects?
- [ ] Data is versioned?
- [ ] Can reproduce any past experiment?

---

## 7.3 Model Deployment - Exercises

### Must-Implement ✓

**Deployment Patterns**
- [ ] REST API (FastAPI)
- [ ] Containerization (Docker)
- [ ] Orchestration (Kubernetes)
- [ ] Serverless option
- [ ] Batch prediction pipeline

### Coding Exercises 💻

**CE3.1**: REST API for model serving
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model at startup
model = joblib.load("model.pkl")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # YOUR CODE HERE
    # Input validation
    # Preprocessing
    # Prediction
    # Logging
    pass

@app.get("/health")
async def health():
    # Health check endpoint
    pass

# Add:
# - Authentication
# - Rate limiting
# - Error handling
# - Logging
```

**CE3.2**: Dockerize ML service
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code and model
COPY . .

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# YOUR TASK:
# - Multi-stage build
# - Optimize image size
# - Security best practices
```

**CE3.3**: Kubernetes deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: your-model:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

# YOUR TASK:
# - Add horizontal autoscaling
# - Configure liveness/readiness probes
# - Set up service
# - Ingress configuration
```

**CE3.4**: Batch prediction pipeline
```python
class BatchPredictor:
    def __init__(self, model_path, batch_size=1000):
        self.model = self.load_model(model_path)
        self.batch_size = batch_size

    def predict_from_db(self, query):
        """
        Read data from database in batches
        Make predictions
        Write back to database
        """
        # YOUR CODE HERE
        pass

    def predict_from_files(self, input_path, output_path):
        """
        Read from files (CSV, Parquet, etc.)
        Batch process
        Write predictions
        """
        # YOUR CODE HERE
        pass

# Schedule with Airflow or cron
```

### Applied Problems 🎯

**AP3.1**: Deploy previous project
- Pick a model from earlier phases
- Create REST API
- Dockerize
- Deploy to cloud (AWS/GCP/Azure) or local K8s
- Load test
- Monitor

**AP3.2**: Compare deployment strategies
For the same model:
- Deploy as REST API
- Deploy as serverless function
- Batch prediction job

Measure: latency, cost, scalability, complexity

### Checkpoint 🚦
- [ ] Deployed at least 2 models to production?
- [ ] Can containerize ML applications?
- [ ] Understand orchestration basics?
- [ ] Know when to use each deployment pattern?

---

## 7.4 Monitoring & Observability - Exercises

### Must-Implement ✓

**Monitoring Stack**
- [ ] Application metrics (latency, throughput)
- [ ] Model metrics (predictions, confidence)
- [ ] Data drift detection
- [ ] Alerting system

### Coding Exercises 💻

**CE4.1**: Instrument model serving
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
confidence_gauge = Gauge('prediction_confidence', 'Average confidence')

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()

    # Make prediction
    pred, conf = model.predict(request.features)

    # Record metrics
    prediction_counter.inc()
    prediction_latency.observe(time.time() - start_time)
    confidence_gauge.set(conf)

    return {"prediction": pred, "confidence": conf}

# Set up Prometheus + Grafana dashboard
```

**CE4.2**: Data drift detection
```python
class DriftDetector:
    def __init__(self, reference_data):
        self.reference_stats = self.compute_stats(reference_data)

    def compute_stats(self, data):
        """Compute statistical properties of data"""
        # YOUR CODE HERE
        # Mean, std, distribution, correlations
        pass

    def detect_drift(self, new_data, threshold=0.05):
        """
        Detect if new data has drifted from reference

        Methods:
        - Statistical tests (KS test, chi-square)
        - Population Stability Index (PSI)
        - KL divergence
        """
        # YOUR CODE HERE
        pass

    def alert_if_drift(self, new_data):
        """Send alert if significant drift detected"""
        # YOUR CODE HERE
        pass

# Integrate into serving pipeline
```

**CE4.3**: Model performance monitoring
```python
class ModelMonitor:
    def track_prediction(self, input_features, prediction, actual=None):
        """
        Log predictions for monitoring

        Track:
        - Prediction distribution
        - Confidence scores
        - Actual labels (when available)
        - Performance metrics
        """
        # YOUR CODE HERE
        pass

    def compute_metrics(self, window='1d'):
        """
        Compute model performance over time window
        """
        # YOUR CODE HERE
        pass

    def detect_degradation(self):
        """
        Detect if model performance is degrading
        """
        # YOUR CODE HERE
        pass
```

### Applied Problems 🎯

**AP4.1**: Build monitoring dashboard
- Set up Prometheus + Grafana
- Monitor deployed model
- Track: latency, throughput, error rate, prediction distribution
- Set up alerts for anomalies

**AP4.2**: Implement drift detection
- Collect production data
- Compare with training data
- Detect and alert on drift
- Trigger retraining pipeline

### Checkpoint 🚦
- [ ] Deployed model has comprehensive monitoring?
- [ ] Alerts set up for failures and drift?
- [ ] Dashboard shows key metrics?
- [ ] Can detect model degradation?

---

## 7.5 CI/CD for ML - Exercises

### Must-Implement ✓

**ML Pipeline Automation**
- [ ] Automated testing (data, model, code)
- [ ] Continuous training triggers
- [ ] Automated deployment
- [ ] Rollback mechanisms

### Coding Exercises 💻

**CE5.1**: Testing ML systems
```python
# tests/test_data.py
def test_data_schema():
    """Test that data matches expected schema"""
    # YOUR CODE HERE
    pass

def test_data_quality():
    """Test for missing values, outliers, etc."""
    # YOUR CODE HERE
    pass

# tests/test_model.py
def test_model_output_range():
    """Test that predictions are in valid range"""
    # YOUR CODE HERE
    pass

def test_model_invariances():
    """Test model behavior on edge cases"""
    # YOUR CODE HERE
    pass

def test_model_performance():
    """Test model achieves minimum performance on test set"""
    # YOUR CODE HERE
    pass

# tests/test_api.py
def test_api_endpoints():
    """Test API endpoints"""
    # YOUR CODE HERE
    pass

def test_api_latency():
    """Test API response time"""
    # YOUR CODE HERE
    pass
```

**CE5.2**: GitHub Actions for ML
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
      - name: Test data quality
        run: python scripts/test_data.py
      - name: Test model performance
        run: python scripts/test_model.py

  train:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Train model
        run: python train.py
      - name: Evaluate model
        run: python evaluate.py
      - name: Upload model
        # Upload to model registry

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t model:${{ github.sha }} .
      - name: Push to registry
        # Push to container registry
      - name: Deploy to staging
        # Deploy to staging environment
      - name: Run smoke tests
        # Test staging deployment
      - name: Deploy to production
        # Gradual rollout
```

### Applied Problems 🎯

**AP5.1**: Full CI/CD pipeline
For a previous project:
- Write comprehensive tests
- Set up GitHub Actions
- Automated training on data changes
- Automated deployment on model improvements
- Rollback mechanism

### Checkpoint 🚦
- [ ] ML projects have automated testing?
- [ ] CI/CD pipeline set up?
- [ ] Can deploy with confidence?
- [ ] Rollback tested?

---

## 7.6 ML Platform Engineering - Exercises

### Advanced Projects 🏗️

**AP6.1**: Feature Store
Build a simple feature store:
- Online and offline feature serving
- Point-in-time correct joins
- Feature versioning
- Python SDK

**AP6.2**: Model Registry
- Centralized model storage
- Versioning and lineage
- Promotion workflow (dev → staging → prod)
- Metadata tracking

**AP6.3**: Experiment Platform
- Self-service model training
- Standardized pipelines
- Resource management
- Cost tracking

---

## Final Phase 7 Checkpoint 🎯

### Comprehensive Project: Production ML System

Build an end-to-end production ML system with ALL best practices:

**System Components:**

1. **Data Pipeline**
   - Automated data ingestion
   - Data validation and quality checks
   - Feature engineering
   - Data versioning (DVC)

2. **Training Pipeline**
   - Experiment tracking (MLflow)
   - Hyperparameter tuning
   - Model validation
   - Model registry

3. **Deployment**
   - REST API (FastAPI)
   - Docker containerization
   - Kubernetes deployment
   - Auto-scaling
   - Blue-green or canary deployment

4. **Monitoring**
   - Application metrics (Prometheus)
   - Model metrics
   - Data drift detection
   - Dashboards (Grafana)
   - Alerting

5. **CI/CD**
   - Automated testing (data, model, API)
   - GitHub Actions or GitLab CI
   - Automated training on schedule
   - Automated deployment
   - Rollback capability

6. **Infrastructure as Code**
   - Terraform or CloudFormation
   - Version controlled
   - Reproducible

7. **Documentation**
   - Architecture diagrams
   - API documentation
   - Runbooks
   - Onboarding guide

**Success Criteria:**
- Handles 100+ requests/second
- <100ms p99 latency
- Automated end-to-end
- Comprehensive monitoring
- Zero-downtime deployments
- Fully documented
- Production-ready code quality

**Time estimate**: 6-8 weeks

### Knowledge Self-Assessment

You've mastered MLOps if:

**System Design:**
- [ ] Can design production ML systems end-to-end
- [ ] Understand all components and tradeoffs
- [ ] Can choose appropriate architectures

**Implementation:**
- [ ] Deployed multiple models to production
- [ ] Set up comprehensive monitoring
- [ ] Implemented CI/CD for ML
- [ ] Built data pipelines

**Operations:**
- [ ] Can debug production ML issues
- [ ] Handle model degradation
- [ ] Manage model versioning
- [ ] Optimize costs

**Best Practices:**
- [ ] Testing ML systems comprehensively
- [ ] Infrastructure as code
- [ ] Documentation and runbooks
- [ ] Incident response

**Your SWE Skills Applied:**
- [ ] Clean, tested, production-quality code
- [ ] Proper software architecture
- [ ] DevOps practices
- [ ] System observability

### Quick Self-Test (1 hour)

1. Design a real-time recommendation system (architecture diagram)
2. Write a GitHub Actions workflow for ML CI/CD
3. How would you detect and handle model drift?
4. Implement health check endpoint for ML API
5. Design monitoring strategy for deployed model

### Career Readiness

If you completed Phase 7, you're ready for:
- **ML Engineer** roles
- **ML Platform Engineer** roles
- **Applied AI Engineer** roles
- **Research Engineer** roles (especially at scale)

Your combination of ML knowledge + production engineering is rare and valuable!

---

## Congratulations! 🎉

If you've made it through all 7 phases with comprehensive projects, you've transformed from SWE to ML Engineer with deep expertise in:

✅ Statistics & probability (inferential + Bayesian)
✅ Mathematical foundations (linear algebra, optimization)
✅ Classical ML algorithms
✅ Deep learning (from scratch to modern architectures)
✅ Modern AI (LLMs, RAG, agents)
✅ Production ML systems (your competitive advantage)

**What's next?**
- Specialize in a domain (NLP, computer vision, reinforcement learning)
- Contribute to open-source ML projects
- Research and publish
- Build AI products
- Share your knowledge (blog, teach, speak)

You've built a strong foundation. Keep learning, building, and pushing boundaries! 🚀
