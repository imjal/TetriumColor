#!/bin/bash
# Quick test of the observer simulation script

echo "Testing observer simulation script..."
echo ""

# Activate conda environment
conda init
conda activate tetrium

# Run the simulation with a test image
python scripts/simulate_observer_views.py \
    measurements/2025-11-4/tests_noise_0.0_scone_noise_0.1/genetic_color_picker_scone_noise \
    --primaries-dir measurements/2025-10-11/primaries/ \
    --num-observers 8 \
    --create-comparison \
    --debug \
    --output-dir measurements/2025-11-4/observer-simulations/

echo ""
echo "Test complete! Check test_outputs/observer_simulation_test/ for results"

