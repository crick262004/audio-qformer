#!/bin/bash
# Monitor the quick test job

JOB_ID=1592
LOG_OUT="logs/quick_test_${JOB_ID}.out"
LOG_ERR="logs/quick_test_${JOB_ID}.err"

echo "=========================================="
echo "Monitoring Quick Test Job: $JOB_ID"
echo "=========================================="
echo ""

# Check job status
echo "Job Status:"
squeue -u $USER | grep $JOB_ID || echo "Job not found in queue (may have completed)"
echo ""

# Show last lines of output
echo "=========================================="
echo "Last 30 lines of output log:"
echo "=========================================="
tail -30 "$LOG_OUT" 2>/dev/null || echo "Output log not found yet"
echo ""

# Show last lines of error log (where training info goes)
echo "=========================================="
echo "Last 40 lines of error log (training info):"
echo "=========================================="
tail -40 "$LOG_ERR" 2>/dev/null || echo "Error log not found yet"
echo ""

# Check for completion
if [ -f "$LOG_OUT" ]; then
    if grep -q "Quick test completed!" "$LOG_OUT"; then
        echo "=========================================="
        echo "✓ TEST COMPLETED SUCCESSFULLY!"
        echo "=========================================="
        echo ""
        echo "Check results in: ./qformer_quick_test_run"
        ls -lh qformer_quick_test_run/ 2>/dev/null || echo "Output directory not found"
    fi
fi
