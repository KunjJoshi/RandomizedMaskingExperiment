#!/bin/bash
# deduplicate_enron.sh
# Full pipeline for deduplicating Enron dataset using Lee et al.'s method

set -e  # Exit on any error

# ============================================================
# Configuration
# ============================================================
WORK_DIR="../datasets/enron"
DEDUP_REPO_DIR="../deduplicate-text-datasets"
MIN_LENGTH=100  # Minimum duplicate length in bytes (50 tokens = ~100 bytes)
NUM_CORES=8     # Number of CPU cores to use

# File paths
DOC_DEDUP_CSV="$WORK_DIR/deduplicated_docs_enron_training_split.csv"
TXT_FILE="$WORK_DIR/enron_emails_docdedup.txt"
DEDUP_TXT_FILE="$WORK_DIR/enron_emails_full_dedup.txt"
FINAL_CSV="$WORK_DIR/deduplicated_train_split.csv"

# ============================================================
# Step 0: Setup
# ============================================================
echo "============================================================"
echo "Enron Email Deduplication Pipeline"
echo "============================================================"
echo "Min duplicate length: $MIN_LENGTH bytes"
echo "CPU cores: $NUM_CORES"
echo ""

# Check if deduplicate-text-datasets exists
if [ ! -d "$DEDUP_REPO_DIR" ]; then
    echo "ERROR: deduplicate-text-datasets not found at $DEDUP_REPO_DIR"
    echo "Please clone it first:"
    echo "  git clone https://github.com/google-research/deduplicate-text-datasets"
    exit 1
fi

# Create work directory
mkdir -p "$WORK_DIR"

# ============================================================
# Step 1: Document-Level Deduplication
# ============================================================
echo "Step 1: Running document-level deduplication (MinHash)..."
python3 deduplicate_document.py

if [ $? -ne 0 ]; then
    echo "ERROR: Document-level deduplication failed"
    exit 1
fi

# Verify output exists
if [ ! -f "$DOC_DEDUP_CSV" ]; then
    echo "ERROR: Expected output file not found: $DOC_DEDUP_CSV"
    exit 1
fi

echo "✓ Document-level deduplication complete"
echo "  Output: $DOC_DEDUP_CSV"
echo ""

# ============================================================
# Step 2: Convert CSV to Text File (one email per line)
# ============================================================
echo "Step 2: Converting CSV to text file..."
python3 prepare_docdedup.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to convert CSV to text"
    exit 1
fi

# Verify output exists
if [ ! -f "$TXT_FILE" ]; then
    echo "ERROR: Expected text file not found: $TXT_FILE"
    exit 1
fi

echo "✓ CSV converted to text file"
echo "  Output: $TXT_FILE"
echo ""

# ============================================================
# Step 3: Build Rust Deduplicator (if not already built)
# ============================================================
echo "Step 3: Building deduplication tool..."
cd "$DEDUP_REPO_DIR"

if [ ! -f "target/debug/dedup_dataset" ]; then
    echo "Building Rust binary..."
    cargo build
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build Rust binary"
        exit 1
    fi
else
    echo "Binary already exists, skipping build"
fi

echo "✓ Deduplication tool ready"
echo ""

# ============================================================
# Step 4: Create tmp directory and set file limits
# ============================================================
echo "Step 4: Setting up environment..."
mkdir -p tmp
ulimit -Sn 100000
echo "✓ Environment configured"
echo ""

# ============================================================
# Step 5: Run ExactSubstr Deduplication
# ============================================================
echo "Step 5: Running ExactSubstr deduplication (this may take a while)..."
echo "Finding duplicate substrings ≥ $MIN_LENGTH bytes..."

bash scripts/deduplicate_single_file.sh \
    "$TXT_FILE" \
    "$DEDUP_TXT_FILE" \
    "$MIN_LENGTH" \
    "$NUM_CORES"

if [ $? -ne 0 ]; then
    echo "ERROR: ExactSubstr deduplication failed"
    exit 1
fi

# Verify output exists
if [ ! -f "$DEDUP_TXT_FILE" ]; then
    echo "ERROR: Expected deduplicated text file not found: $DEDUP_TXT_FILE"
    exit 1
fi

echo "✓ ExactSubstr deduplication complete"
echo "  Output: $DEDUP_TXT_FILE"
echo ""

# ============================================================
# Step 6: Convert Back to CSV
# ============================================================
echo "Step 6: Converting deduplicated text back to CSV..."
cd - > /dev/null  # Return to original directory

python3 back_to_csv.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to convert back to CSV"
    exit 1
fi

# Verify final output exists
if [ ! -f "$FINAL_CSV" ]; then
    echo "ERROR: Expected final CSV not found: $FINAL_CSV"
    exit 1
fi

echo "✓ Final CSV file created"
echo "  Output: $FINAL_CSV"
echo ""

# ============================================================
# Step 7: Summary Statistics
# ============================================================
echo "============================================================"
echo "DEDUPLICATION COMPLETE!"
echo "============================================================"
echo ""
echo "Pipeline stages:"
echo "  1. Document-level dedup → $DOC_DEDUP_CSV"
echo "  2. Text conversion       → $TXT_FILE"
echo "  3. ExactSubstr dedup     → $DEDUP_TXT_FILE"
echo "  4. Final CSV             → $FINAL_CSV"
echo ""
echo "Intermediate files kept for inspection."
echo ""
echo "Next steps:"
echo "  1. Inspect $FINAL_CSV"
echo "  2. Use for RMFT training"
echo ""