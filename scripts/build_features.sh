#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Lab 5: Feature Engineering Pipeline${NC}"
echo "======================================"

# Check prerequisites
if [ ! -f "data/ratings_clean.csv" ]; then
    echo -e "${RED}Error: data/ratings_clean.csv not found${NC}"
    echo "Please run Lab 4 (data ingestion) first"
    exit 1
fi

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run feature engineering
echo -e "${BLUE}Running feature engineering pipeline...${NC}"
python src/prepare_features.py data/ratings_clean.csv models

# Verify output
if [ -f "models/rating_features.pkl" ]; then
    echo -e "${GREEN}✓ Feature store created successfully${NC}"
    ls -lh models/rating_features.pkl
else
    echo -e "${RED}Error: Feature store not created${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Pipeline complete!${NC}"
```

Make it executable:
```bash
chmod +x scripts/build_features.sh