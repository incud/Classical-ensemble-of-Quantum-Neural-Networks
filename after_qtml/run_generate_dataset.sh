echo "Regenerating datasets..."
rm -vr datasets || true
mkdir datasets
python3.9 generate_dataset.py run
