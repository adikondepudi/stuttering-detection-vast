# Stuttering Detection Training on Vast.ai

This guide explains how to run the stuttering detection training pipeline on Vast.ai cloud computing platform.

## 🚀 Quick Start

### 1. Set up Vast.ai Account
- Create account at [vast.ai](https://vast.ai)
- Add billing method and credits ($5 minimum)
- Upload SSH key (optional but recommended)

### 2. Rent a GPU Instance
- Select PyTorch template from Templates tab
- Filter for GPU with:
  - **GPU Memory**: ≥8GB (12GB+ recommended)
  - **CUDA Version**: 11.7 or newer
  - **Disk Space**: ≥50GB
  - **Internet Speed**: ≥100 Mbps
- Rent the instance

### 3. Upload Your Code
```bash
# Option 1: Using Jupyter (recommended for beginners)
# Click "Open" button when instance is ready
# Upload files through Jupyter interface

# Option 2: Using SCP/SFTP
scp -r . root@<instance-ip>:/workspace/

# Option 3: Using git
git clone <your-repository-url>
```

### 4. Configure Your Data
Edit `config/config.yaml`:
```yaml
# Enable cloud storage and set your data source
cloud:
  enabled: true
  type: "gdrive"  # or "s3", "gcs"
  gdrive_file_id: "your_actual_file_id_here"
```

### 5. Start Training
```bash
# Quick start (recommended)
bash start_training.sh

# Or manual start
python setup_vast.py
python main.py --config config/config.yaml --mode all --use-wandb
```

## 📁 File Structure

### New Files for Vast.ai
```
├── setup_vast.py           # Environment setup script
├── start_training.sh        # Persistent training starter
├── monitor_training.py      # Training progress monitor
├── README_VAST.md          # This file
└── requirements.txt        # Updated with cloud dependencies
```

### Modified Files
```
├── config/config.yaml      # Cloud-optimized configuration
├── main.py                # Enhanced with cloud features
├── src/
│   ├── data_preprocessing.py   # Cloud storage support
│   ├── train.py               # Cost optimization & monitoring
│   └── utils.py              # Resource monitoring utilities
```

## 🎛️ Configuration Options

### Cloud Storage Configuration
```yaml
cloud:
  enabled: true
  type: "gdrive"  # Options: gdrive, s3, gcs
  
  # Google Drive
  gdrive_file_id: "your_file_id"
  
  # AWS S3
  # s3_bucket: "your-bucket"
  # s3_key: "path/to/data.zip"
  
  # Google Cloud Storage
  # gcs_bucket: "your-bucket"
  # gcs_blob: "path/to/data.zip"
```

### Cost Optimization Settings
```yaml
cost_optimization:
  max_training_hours: 8.0        # Stop after 8 hours
  target_performance: 0.80       # Stop early if F1 > 0.80
  enable_early_cost_stop: true   # Enable cost-based stopping

training:
  batch_size: 16                 # Reduced for GPU memory
  max_epochs: 50                 # Reduced for cost efficiency
  save_checkpoint_every: 5       # Save every 5 epochs
```

### Monitoring Configuration
```yaml
monitoring:
  use_wandb: true                # Enable Weights & Biases
  wandb_project: "stuttering-detection"
  print_resource_usage: true     # Monitor GPU/CPU usage
```

## 🔧 Usage Commands

### Basic Training
```bash
# Full pipeline (preprocessing + training)
python main.py --mode all

# Only preprocessing
python main.py --mode preprocess

# Only training (requires preprocessed data)
python main.py --mode train

# With Weights & Biases logging
python main.py --mode all --use-wandb
```

### Resume Training
```bash
# Resume from specific checkpoint
python main.py --mode train --resume checkpoints/checkpoint_epoch_15.pth

# Resume from best model
python main.py --mode train --resume checkpoints/best_model.pth
```

### Persistent Training (Recommended)
```bash
# Start training in tmux session
bash start_training.sh

# Check tmux sessions
tmux list-sessions

# Attach to training session
tmux attach-session -t training_session

# Detach from session (keep training running)
# Press: Ctrl+B, then D
```

### Monitor Training Progress
```bash
# Check status once
python monitor_training.py --once

# Continuous monitoring
python monitor_training.py --interval 60

# Generate training plots
python monitor_training.py --plot --once
```

## 📊 Data Preparation

### Supported Data Sources

#### Google Drive
1. Upload your dataset as a ZIP file to Google Drive
2. Get the file ID from the sharing URL:
   - URL: `https://drive.google.com/file/d/1ABC123DEF456/view`
   - File ID: `1ABC123DEF456`
3. Set in config: `gdrive_file_id: "1ABC123DEF456"`

#### AWS S3
```yaml
cloud:
  type: "s3"
  s3_bucket: "my-dataset-bucket"
  s3_key: "stuttering-data/dataset.zip"
```

#### Google Cloud Storage
```yaml
cloud:
  type: "gcs"
  gcs_bucket: "my-dataset-bucket"
  gcs_blob: "stuttering-data/dataset.zip"
```

### Dataset Format
Your ZIP file should contain:
```
dataset.zip
├── audio_file1.wav
├── audio_file2.wav
├── annotations.csv
└── ...
```

Annotations CSV format:
```csv
audio_file,start_time,end_time,disfluency_type,annotator_id
audio_file1.wav,1.2,2.5,Prolongation,annotator1
audio_file1.wav,3.1,3.8,Word Repetition,annotator1
```

## 💰 Cost Management

### Typical Costs (Approximate)
- **RTX 3090 (24GB)**: $0.20-0.40/hour
- **RTX 4090 (24GB)**: $0.30-0.50/hour
- **A40 (48GB)**: $0.50-0.80/hour
- **A100 (80GB)**: $1.00-2.00/hour

### Cost Optimization Tips
1. **Start Small**: Test with small batch sizes first
2. **Use Spot Instances**: Cheaper but can be interrupted
3. **Set Time Limits**: Configure `max_training_hours`
4. **Early Stopping**: Set reasonable `target_performance`
5. **Monitor Progress**: Use `monitor_training.py` to check remotely
6. **Clean Up**: Destroy instances when done

### Automatic Cost Controls
The pipeline includes built-in cost protection:
- Training stops after `max_training_hours`
- Early stopping if target performance is reached
- Automatic checkpoint saving to prevent data loss
- Resource monitoring to optimize usage

## 🔍 Monitoring & Debugging

### Check Training Status
```bash
# View current status
python monitor_training.py --once

# Continuous monitoring every 30 seconds
python monitor_training.py

# Generate and save training plots
python monitor_training.py --plot --once
```

### Access Training Session
```bash
# List tmux sessions
tmux list-sessions

# Attach to training session
tmux attach-session -t training_session

# View training logs in real-time
tail -f logs/training.log  # if logging to file
```

### Check Resource Usage
```bash
# GPU usage
nvidia-smi

# System resources
htop

# Python resource monitoring
python -c "from src.utils import monitor_resources; monitor_resources()"
```

### Common Issues

#### Out of Memory Errors
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 8  # Reduce from 16
```

#### Slow Training
- Check GPU utilization with `nvidia-smi`
- Ensure data loading isn't bottleneck
- Consider reducing model size or sequence length

#### Connection Issues
- Use tmux/screen for persistent sessions
- Set up SSH keep-alive in `~/.ssh/config`:
```
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

## 📈 Results & Outputs

### Training Outputs
```
checkpoints/
├── best_model.pth              # Best model checkpoint
├── checkpoint_epoch_X.pth      # Periodic checkpoints
├── training_metrics.json       # Training history
└── test_results.json          # Final evaluation results
```

### Download Results
```bash
# Using SCP
scp -r root@<instance-ip>:/workspace/checkpoints/ ./results/

# Or download through Jupyter interface
```

### Weights & Biases Dashboard
If WandB is enabled, view results at: https://wandb.ai/your-username/stuttering-detection

## 🏁 Complete Workflow Example

```bash
# 1. Set up environment
python setup_vast.py

# 2. Configure data source (edit config/config.yaml)
# Set your gdrive_file_id or other cloud storage settings

# 3. Start persistent training
bash start_training.sh

# 4. Monitor progress (in another terminal/session)
python monitor_training.py --interval 60 --plot

# 5. Download results when complete
# Results will be in checkpoints/ directory
```

## 🆘 Troubleshooting

### Training Won't Start
1. Check data configuration in `config/config.yaml`
2. Verify cloud storage credentials
3. Run `python setup_vast.py` to check environment

### Training Stops Unexpectedly
1. Check if cost limits were reached
2. Look for out-of-memory errors
3. Verify instance wasn't preempted (spot instances)

### Can't Access Results
1. Download through Jupyter interface
2. Use SCP/SFTP to transfer files
3. Check WandB dashboard if enabled

### Need Help
1. Check logs in tmux session: `tmux attach-session -t training_session`
2. Monitor resource usage: `python monitor_training.py --once`
3. Review configuration: Ensure all paths and IDs are correct

## 📝 Best Practices

1. **Always use tmux/screen** for long-running training
2. **Set reasonable time limits** to avoid unexpected costs
3. **Monitor progress remotely** instead of keeping connection open
4. **Save checkpoints frequently** in case of interruption
5. **Test with small datasets first** before full training
6. **Keep backups** of your best models
7. **Document your experiments** using WandB or similar tools

---

**Happy Training! 🚀**

For more help, refer to the [Vast.ai documentation](https://vast.ai/docs/) or the original training pipeline documentation.