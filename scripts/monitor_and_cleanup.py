import paramiko
import os
import time
import requests
import stat
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Monitor")
logging.getLogger("paramiko").setLevel(logging.WARNING)

# Load RunPod API Key from environment or optional .env file
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    # Try reading from .env file manually
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if line.strip().startswith("RUNPOD_API_KEY="):
                    RUNPOD_API_KEY = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    break

if not RUNPOD_API_KEY:
    logger.error("RUNPOD_API_KEY not found in environment or .env file!")
    sys.exit(1)

POD_ID = "dt9e8sndzq5j5g"
POD_IP = "70.51.183.147"
POD_PORT = 24042

def is_training_active(ssh):
    stdin, stdout, stderr = ssh.exec_command("ps -ef | grep train_engine.py | grep -v grep | grep -v bash | grep -v nohup")
    output = stdout.read().decode('utf-8', errors='ignore').strip()
    return len(output) > 0

def get_active_log_file(ssh):
    # Check if Small model training is running (excluding bash/nohup wrappers)
    stdin, stdout, stderr = ssh.exec_command("ps -ef | grep 'variant s' | grep -v grep | grep -v bash | grep -v nohup")
    s_running = len(stdout.read().decode('utf-8', errors='ignore').strip()) > 0
    if s_running:
        return "/workspace/CIRCA/train_engine_s.log", "YOLOv12-Small"
        
    # Check if Medium model training is running (excluding bash/nohup wrappers)
    stdin, stdout, stderr = ssh.exec_command("ps -ef | grep 'variant m' | grep -v grep | grep -v bash | grep -v nohup")
    m_running = len(stdout.read().decode('utf-8', errors='ignore').strip()) > 0
    if m_running:
        return "/workspace/CIRCA/train_engine_m.log", "YOLOv12-Medium"
        
    # Fallback/Default
    return None, None

def sftp_download_dir(sftp, remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    try:
        entries = sftp.listdir_attr(remote_dir)
    except IOError:
        return
        
    for entry in entries:
        remote_path = remote_dir + "/" + entry.filename
        local_path = os.path.join(local_dir, entry.filename)
        
        if stat.S_ISDIR(entry.st_mode):
            sftp_download_dir(sftp, remote_path, local_path)
        else:
            logger.info(f"Downloading {entry.filename}...")
            sftp.get(remote_path, local_path)

def terminate_pod():
    # Use official runpod SDK to terminate
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    try:
        runpod.terminate_pod(POD_ID)
        logger.info(f"Pod {POD_ID} terminated successfully via SDK.")
    except Exception as e:
        logger.error(f"Failed to terminate pod via SDK: {e}")

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = paramiko.RSAKey.from_private_key_file(os.path.expanduser('~/.ssh/id_rsa'))
    
    logger.info("==========================================================")
    logger.info("  CIRCA RunPod Real-Time Training Monitor & Auto-Cleanup ")
    logger.info("==========================================================")
    logger.info(f"Pod ID: {POD_ID} | Host: {POD_IP}:{POD_PORT}")
    logger.info("Connecting...")
    
    # Store length of processed log to print only new content
    last_log_size = 0
    
    while True:
        try:
            ssh.connect(POD_IP, port=POD_PORT, username='root', pkey=pkey, timeout=10)
            
            # Check if CIRCA workspace has been extracted
            stdin, stdout, stderr = ssh.exec_command("[ -d '/workspace/CIRCA' ] && echo 'yes' || echo 'no'")
            extracted = stdout.read().decode('utf-8', errors='ignore').strip() == 'yes'
            
            if not extracted:
                logger.info("Workspace /workspace/CIRCA not found. Waiting for upload and extraction to complete...")
                ssh.close()
                time.sleep(30)
                continue
                
            # Check training activity
            active = is_training_active(ssh)
            log_file, model_name = get_active_log_file(ssh)
            
            if active and log_file:
                # Read training logs dynamically
                stdin, stdout, stderr = ssh.exec_command(f"wc -c {log_file}")
                try:
                    size_out = stdout.read().decode('utf-8', errors='ignore').strip()
                    current_size = int(size_out.split()[0])
                except Exception:
                    current_size = 0
                
                if current_size > last_log_size:
                    # Fetch only the new content
                    bytes_to_read = current_size - last_log_size
                    # Limit to 5000 bytes per chunk to avoid SSH buffer bloat
                    bytes_to_read = min(bytes_to_read, 5000)
                    
                    # Read using tail or dd
                    cmd = f"tail -c {bytes_to_read} {log_file}"
                    stdin_log, stdout_log, stderr_log = ssh.exec_command(cmd)
                    new_text = stdout_log.read().decode('utf-8', errors='ignore').encode('ascii', errors='ignore').decode('ascii')
                    
                    if new_text.strip():
                        print(f"\n--- [{model_name} LOG UPDATE] ---")
                        print(new_text.strip())
                        print("-" * 40)
                    
                    last_log_size = current_size
                else:
                    # Just print simple heartbeat if no new text
                    sys.stdout.write(".")
                    sys.stdout.flush()
            else:
                # If training is inactive, check if logs show completed runs
                # Or wait if setup is still running
                stdin, stdout, stderr = ssh.exec_command("ps -ef | grep setup-runpod-environment.sh | grep -v grep")
                setup_running = len(stdout.read().decode('utf-8', errors='ignore').strip()) > 0
                
                if setup_running:
                    logger.info("Setup script is still running. Waiting for training to start...")
                    # Print setup logs
                    stdin, stdout, stderr = ssh.exec_command("tail -n 3 /workspace/CIRCA/setup_runpod.log")
                    print(f"  Setup > {stdout.read().decode('utf-8', errors='ignore').strip()}")
                else:
                    logger.info("Training is INACTIVE. Checking completed runs directories...")
                    
                    # Connect and check runs directory
                    sftp = ssh.open_sftp()
                    
                    # Download Small run results
                    remote_s_dir = "/workspace/CIRCA/runs/detect/CIRCA_V12S_008_TRAIN_copypaste_small"
                    local_s_dir = "d:/FYP/CIRCA/runs/detect/CIRCA_V12S_008_TRAIN_copypaste_small"
                    logger.info(f"Downloading YOLOv12-Small training results...")
                    sftp_download_dir(sftp, remote_s_dir, local_s_dir)
                    
                    # Download Medium run results
                    remote_m_dir = "/workspace/CIRCA/runs/detect/CIRCA_V12M_009_TRAIN_copypaste_medium"
                    local_m_dir = "d:/FYP/CIRCA/runs/detect/CIRCA_V12M_009_TRAIN_copypaste_medium"
                    logger.info(f"Downloading YOLOv12-Medium training results...")
                    sftp_download_dir(sftp, remote_m_dir, local_m_dir)
                    
                    sftp.close()
                    ssh.close()
                    
                    logger.info("Results downloaded successfully to local runs/ directory!")
                    
                    # Terminate the pod to save costs
                    logger.info("Terminating RunPod instance...")
                    terminate_pod()
                    break
            
            ssh.close()
        except Exception as e:
            logger.error(f"Error during status check: {e}")
            
        # Poll every 15 seconds for real-time responsiveness
        time.sleep(15)

if __name__ == "__main__":
    main()
