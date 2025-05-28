# Vast.ai Deployment Guide

## 📋 File Organization Summary

Here's exactly what you need to do to deploy your stuttering detection pipeline on Vast.ai:

### 1. FILES TO ADD (New Files)
Place these new files in your project root directory:

- `setup_vast.py` - Environment setup script
- `start_training.sh` - Persistent training starter (make executable: `chmod +x start_training.sh`)
- `monitor_training.py` - Training progress monitor
- `README_VAST.md` - Vast.ai specific documentation
- `DEPLOYMENT_GUIDE.md` - This guide

### 2. FILES TO REPLACE (Modified Files)
Replace these existing files with the updated versions:

- `requirements.txt` - Updated with cloud dependencies
- `config/config.yaml` - Cloud-optimized configuration
- `main.py` - Enhanced with cloud features and monitoring
- `src/data_preprocessing.py` - Added cloud storage support
- `src/train.py` - Added cost optimization and monitoring
- `src/utils.py` - Added resource monitoring utilities
- `src/__init__.py` - Updated imports

### 3. FILES TO KEEP UNCHANGED
These files remain the same:

- `src/feature_extraction.py`
- `src/model.py`
- `src/dataset.py`

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Local Files
1. Download all the provided artifacts
2. Replace the existing files with updated versions
3. Add the new files to your project directory
4. Make the shell script executable:
   ```bash
   chmod +x start_training.sh
   ```

### Step 2: Prepare Your Data
1. Create a ZIP file containing:
   - Audio files (.wav, .mp3, etc.)
   - Annotation CSV files
   - Any additional data

2. Upload to cloud storage:
   - **Google Drive**: Upload ZIP, get shareable link, extract file ID
   - **AWS S3**: Upload to S3 bucket
   - **Google Cloud Storage**: Upload to GCS bucket

### Step 3: Set Up Vast.ai Account
1. Go to [vast.ai](https://vast.ai)
2. Create account and add billing method
3. Add at least $5 in credits
4. (Optional) Upload SSH public key for secure access

### Step 4: Rent GPU Instance
1. Go to "Templates" tab
2. Select a PyTorch template
3. Filter for:
   - GPU Memory: ≥8GB (12GB+ recommended)
   - CUDA Version: 11.7+
   - Disk Space: ≥50GB
   - Good internet speed for downloads
4. Click "RENT" on suitable instance

### Step 5: Upload Your Code
**Option A: Using Jupyter (Easiest)**
1. Wait for instance to show "Open" button
2. Click "Open" to access Jupyter
3. Upload all your files through Jupyter interface

**Option B: Using SCP/SSH**
```bash
# Get instance IP from Vast.ai dashboard
scp -r your_project_folder root@<instance-ip>:/workspace/
```

**Option C: Using Git**
```bash
# SSH into instance
ssh root@<instance-ip>
cd /workspace
git clone <your-repository-url>
```

### Step 6: Configure Your Data Source
Edit `config/config.yaml` on the remote instance:

```yaml
# Enable cloud storage
cloud:
  enabled: true
  type: "gdrive"  # Change as needed
  gdrive_file_id: "YOUR_ACTUAL_FILE_ID_HERE"  # Replace with your file ID
```

For other storage types:
```yaml
# AWS S3
cloud:
  enabled: true
  type: "s3"
  s3_bucket: "your-bucket-name"
  s3_key: "path/to/your/data.zip"

# Google Cloud Storage  
cloud:
  enabled: true
  type: "gcs"
  gcs_bucket: "your-bucket-name"
  gcs_blob: "path/to/your/data.zip"
```

### Step 7: Start Training
**Quick Start (Recommended):**
```bash
cd /workspace/your_project_folder
bash start_training.sh
```

**Manual Start:**
```bash
cd /workspace/your_project_folder
python setup_vast.py
python main.py --config config/config.yaml --mode all --use-wandb
```

### Step 8: Monitor Training
**From another terminal/session:**
```bash
# Check status once
python monitor_training.py --once

# Continuous monitoring
python monitor_training.py --interval 60

# Generate plots
python monitor_training.py --plot --once
```

**Or attach to tmux session:**
```bash
tmux attach-session -t training_session
# Detach with: Ctrl+B, then D
```

### Step 9: Download Results
When training completes, download your results:

**Through Jupyter:**
- Navigate to `checkpoints/` folder
- Download `best_model.pth`, `test_results.json`, etc.

**Through SCP:**
```bash
scp -r root@<instance-ip>:/workspace/your_project_folder/checkpoints/ ./results/
```

### Step 10: Clean Up
**Important: Don't forget to destroy the instance when done!**
1. Go to "Instances" tab in Vast.ai
2. Click "Destroy" on your instance
3. This prevents ongoing charges

## ⚙️ Configuration Tips

### Memory Optimization
If you get out-of-memory errors:
```yaml
training:
  batch_size: 8  # Reduce from default 16
```

### Cost Control
```yaml
cost_optimization:
  max_training_hours: 4.0      # Limit to 4 hours
  target_performance: 0.75     # Stop early if F1 > 0.75
  enable_early_cost_stop: true
```

### Monitoring Setup
```yaml
monitoring:
  use_wandb: true  # Enable for remote monitoring
  wandb_project: "my-stuttering-project"
  print_resource_usage: true
```

## 🔧 Troubleshooting

### Common Issues and Solutions

**1. "No module named 'src'"**
```bash
# Make sure you're in the right directory
cd /workspace/your_project_folder
python setup_vast.py
```

**2. "File not found" errors**
- Check that all files were uploaded correctly
- Verify file paths in configuration

**3. "CUDA out of memory"**
- Reduce batch size in config.yaml
- Use smaller model if needed

**4. Training stops unexpectedly**
- Check if cost limits were reached
- Look for preemption (spot instances)
- Check tmux session: `tmux attach-session -t training_session`

**5. Can't download data from cloud**
- Verify file IDs and permissions
- Check internet connectivity on instance
- Try manual download to test

### Getting Help

**Check logs:**
```bash
tmux attach-session -t training_session
# OR
python monitor_training.py --once
```

**Check resources:**
```bash
nvidia-smi  # GPU usage
htop        # CPU/RAM usage
df -h       # Disk usage
```

## 📊 Expected Costs

Approximate costs for full training (depends on data size and GPU):

- **RTX 3090**: $1.60-3.20 (4-8 hours at $0.40/hr)
- **RTX 4090**: $2.00-4.00 (4-8 hours at $0.50/hr)  
- **A40**: $3.20-6.40 (4-8 hours at $0.80/hr)

## ✅ Pre-flight Checklist

Before starting training, verify:

- [ ] All files uploaded to Vast.ai instance
- [ ] Data source configured in config.yaml
- [ ] Cloud storage file ID/path is correct
- [ ] Cost limits set appropriately
- [ ] WandB account set up (if using)
- [ ] SSH key uploaded (if using SSH)
- [ ] Adequate credits in Vast.ai account

## 🎯 Success Indicators

Training is working correctly when you see:

1. **Setup phase**: Dependencies installing, directories created
2. **Data download**: Files downloading from cloud storage
3. **Preprocessing**: Audio files being processed
4. **Training start**: Epochs beginning, loss decreasing
5. **Monitoring**: Metrics updating, checkpoints saving

You should see output like:
```
GPU detected: NVIDIA RTX 4090
Model parameters: 2,341,234
Starting training...
Epoch 1/50: Train Loss: 0.456, Val F1: 0.234
```

That's it! You're now ready to train your stuttering detection model on Vast.ai. 🚀