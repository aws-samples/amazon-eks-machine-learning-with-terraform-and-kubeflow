#!/bin/bash

# Script to run terraform destroy with retry logic
# Attempts up to 10 times until successful

MAX_ATTEMPTS=10
ATTEMPT=1
SUCCESS=false

echo "Starting terraform destroy with retry logic (max $MAX_ATTEMPTS attempts)..."

while [ $ATTEMPT -le $MAX_ATTEMPTS ] && [ "$SUCCESS" = false ]; do
    echo "Attempt $ATTEMPT of $MAX_ATTEMPTS..."
    
    if terraform destroy -auto-approve; then
        echo "✅ Terraform destroy succeeded on attempt $ATTEMPT"
        SUCCESS=true
    else
        echo "❌ Terraform destroy failed on attempt $ATTEMPT"
        
        if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
            echo "Waiting 30 seconds before retry..."
            sleep 30
        fi
        
        ATTEMPT=$((ATTEMPT + 1))
    fi
done

if [ "$SUCCESS" = true ]; then
    echo "🎉 Terraform destroy completed successfully!"
    exit 0
else
    echo "💥 Terraform destroy failed after $MAX_ATTEMPTS attempts"
    exit 1
fi