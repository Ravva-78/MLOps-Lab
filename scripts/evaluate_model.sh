#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Lab 7: Model Evaluation Pipeline${NC}"
echo "===================================="

# Check prerequisites
if [ ! -f "models/model.pkl" ]; then
    echo -e "${RED}Error: models/model.pkl not found${NC}"
    echo "Please run Lab 6 (training) first"
    exit 1
fi

if [ ! -f "data/processed/ratings_clean.csv" ]; then
    echo -e "${RED}Error: data/processed/ratings_clean.csv not found${NC}"
    exit 1
fi

# Activate venv if needed
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Run evaluation
echo -e "${BLUE}Running evaluation pipeline...${NC}"
python src/evaluate_main.py \
    --model_path models/model.pkl \
    --metadata_path models/metadata.json \
    --test_path data/processed/ratings_clean.csv \
    --ratings_path data/processed/ratings_clean.csv \
    --n_movies 100 \
    --eval_dir evaluations

# Verify output
if [ -f "evaluations/evaluation_report.json" ]; then
    echo -e "${GREEN}✓ Evaluation report created${NC}"
    echo ""
    echo "Report summary:"
    python -c "
import json
with open('evaluations/evaluation_report.json') as f:
    report = json.load(f)
    metrics = report['rating_prediction']
    coverage = report['coverage']
    baseline = report['baselines']
    print(f'  RMSE:              {metrics[\"rmse\"]:.4f}')
    print(f'  MAE:               {metrics[\"mae\"]:.4f}')
    print(f'  Catalog Coverage:  {coverage[\"coverage_ratio\"]:.2%}')
    print(f'  Best Baseline:     {baseline[\"best_baseline_rmse\"]:.4f}')
" 2>/dev/null || cat evaluations/evaluation_report.json
else
    echo -e "${RED}Error: Evaluation report not created${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Evaluation complete!${NC}"

export PYTHONPATH=$(pwd)

python src/evaluate_main.py \
    --model_path models/model.pkl \
    --metadata_path models/metadata.json \
    --test_path data/processed/ratings_clean.csv \
    --ratings_path data/processed/ratings_clean.csv \
    --n_movies 100 \
    --eval_dir evaluations