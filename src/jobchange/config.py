"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_CSV = DATA_DIR / "aug_train.csv"
TEST_CSV = DATA_DIR / "aug_test.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

TARGET = "target"
ID_COL = "enrollee_id"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns the model needs at inference time.
RAW_FEATURES = [
    "city",
    "city_development_index",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "experience",
    "company_size",
    "company_type",
    "last_new_job",
    "training_hours",
]

# gender is left out on purpose (hiring use case)
EXCLUDED_FEATURES = ["gender"]

EDU_ORDER = ["Primary School", "High School", "Graduate", "Masters", "Phd"]
SIZE_ORDER = ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"]
EXPERIENCE_OPTIONS = ["<1"] + [str(i) for i in range(1, 21)] + [">20"]
LAST_JOB_OPTIONS = ["never", "1", "2", "3", "4", ">4"]
ENROLLMENT_OPTIONS = ["no_enrollment", "Full time course", "Part time course"]
MAJOR_OPTIONS = ["STEM", "Business Degree", "Humanities", "Arts", "Other", "No Major"]
COMPANY_TYPE_OPTIONS = ["Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other"]
