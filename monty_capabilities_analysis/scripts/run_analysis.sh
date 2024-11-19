RESULTS_DIR=()
for dir in monty_capabilities_analysis/results/dmc/*/; do
    # Only add the directory name (without path)
    RESULTS_DIR+=("$(basename "$dir")")
done

# Verify the contents of RESULTS_DIR
echo "Found directories:"
for result in "${RESULTS_DIR[@]}"; do
    echo "$result"
done

for experiment in "${RESULTS_DIR[@]}"; do
    # if there is an error, raise an exception
    python monty_capabilities_analysis/scripts/analyze.py --experiment_name "$experiment" 
    echo "Done with $experiment"
done
