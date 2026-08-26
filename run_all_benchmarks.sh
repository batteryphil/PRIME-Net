#!/bin/bash
source venv/bin/activate

echo "=========================================="
echo "STARTING GRAND EMPIRICAL SWEEP"
echo "=========================================="

echo "1/4: Running Macroeconomics..."
python prime2_macro_test.py

echo "------------------------------------------"
echo "2/4: Running Fluid Dynamics..."
python prime2_airfoil_test.py

echo "------------------------------------------"
echo "3/4: Running Materials Science..."
python prime2_concrete_test.py

echo "------------------------------------------"
echo "4/4: Running Quantum Mechanics..."
python prime2_quantum_test.py

echo "=========================================="
echo "GRAND EMPIRICAL SWEEP COMPLETE"
echo "=========================================="
