#!/bin/bash

# Vast.ai Training Starter Script

echo "=================================================="
echo "STUTTERING DETECTION TRAINING - VAST.AI"
echo "=================================================="

# Check if we're already in tmux
if [ -n "$TMUX" ]; then
    echo "Already in tmux session. Starting training directly..."
    python3 setup_vast.py
    python3 main.py --config config/config.yaml --mode all --use-wandb
else
    echo "Creating persistent tmux session..."
    
    # Kill existing session if it exists
    tmux kill-session -t training_session 2>/dev/null || true
    
    # Create new session and run training
    tmux new-session -d -s training_session -c "$(pwd)" \
        'python3 setup_vast.py && python3 main.py --config config/config.yaml --mode all --use-wandb; echo "Training completed. Press any key to exit."; read -n 1'
    
    echo "Training started in tmux session 'training_session'"
    echo ""
    echo "To monitor progress:"
    echo "  tmux attach-session -t training_session"
    echo ""
    echo "To detach from session: Ctrl+B, then D"
    echo "To monitor without attaching:"
    echo "  python3 monitor_training.py --interval 60"
    echo ""
    echo "Session will remain active even if you disconnect."
fi