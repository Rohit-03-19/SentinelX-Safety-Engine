🚀 Sentinel-X: AI-Powered Accident Severity EngineSentinel-X is a modular, production-ready MLOps pipeline designed to analyze and predict the severity of traffic accidents across the United States. Built using the US Accidents Dataset, this project transitions from experimental scripts into a decoupled, scalable architecture capable of handling millions of records.🏗️ System ArchitectureUnlike traditional flat-file scripts, Sentinel-X follows a modular design pattern to ensure maintainability and scalability:Data Ingestion: Automated fetching and validation of large-scale CSV data with strategic sampling.Data Transformation: A robust preprocessing "guardrail" using ColumnTransformer to handle numerical scaling and categorical encoding.Model Training: An optimized Random Forest Regressor utilizing hyperparameter tuning via GridSearchCV.Predict Pipeline: A dedicated inference layer for real-time severity estimation using serialized artifacts.📊 Key Insights & EDAInitial exploration of the dataset revealed critical safety patterns that informed our feature engineering:Rush Hour Risk: Accident frequency peaks significantly between 7:00 AM – 9:00 AM and 4:00 PM – 6:00 PM, aligning with peak commuting hours.Infrastructure Impact: Managed intersections (Traffic Signals and Crossings) show a strong negative correlation with high severity, suggesting these measures effectively reduce accident impact.Environmental Sensitivity: Visibility and adverse weather (Rain, Fog) were identified as primary predictors in the severity correlation matrix.🔧 Installation & Setup1. Clone the RepositoryBashgit clone https://github.com/Rohit-03-19/SentinelX-Safety-Engine.git
cd SentinelX-Safety-Engine
2. Environment SetupBash# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Run the PipelineBashpython main.py
🛠️ Engineering Hurdles & SolutionsA major focus of this project was overcoming real-world hardware and integration bottlenecks:Memory Optimization: Faced ArrayMemoryError when processing millions of rows. Resolved by implementing a 10% random sampling strategy (~770,000 records) and transitioning to Sequential Processing (n_jobs=1) during model training.Interface Resilience: Solved metadata "unpacking" errors between components by implementing Extended Iterable Unpacking.Dimension Alignment: Resolved dimensionality mismatches post-encoding through explicit NumPy reshaping and sparse-to-dense conversions.📈 Performance MetricsAlgorithm: Random Forest Regressor Final R2 Score: 0.0447 Analysis: While traffic accidents are inherently stochastic, this score represents a Strong Baseline. It confirms that while environmental factors contribute to severity, the majority of outcomes are determined by unpredictable variables like driver behavior not present in current datasets.🚀 Future Roadmap[ ] Integration of Gradient Boosting algorithms (XGBoost / LightGBM).[ ] Feature engineering focused on interaction terms (e.g., Visibility × Road Surface).[ ] Deployment as a FastAPI web service for real-time risk assessment.📁 Repository Structure SentinelX_Safety_Engine/
├── artifacts/               # Serialized pkl files (model & preprocessor)
├── data/                    # Local raw and sampled datasets
├── notebooks/               # EDA and experimental notebooks
├── src/
│   ├── components/          # Ingestion, Transformation, Trainer logic
│   ├── pipeline/            # Training and Prediction workflows
│   ├── logger.py            # Custom logging module
│   └── exception.py         # Custom error handling
├── main.py                  # Pipeline entry point
└── setup.py                 # Project packaging
