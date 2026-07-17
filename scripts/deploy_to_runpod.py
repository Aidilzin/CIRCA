import os
import sys
import time
import runpod
import paramiko
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RunPodDeployer")

# Load RunPod API Key from environment or optional .env file
runpod.api_key = os.environ.get("RUNPOD_API_KEY")
if not runpod.api_key:
    # Try reading from .env file manually
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if line.strip().startswith("RUNPOD_API_KEY="):
                    runpod.api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    break

if not runpod.api_key:
    logger.error("RUNPOD_API_KEY not found in environment or .env file!")
    sys.exit(1)

def get_ssh_connection_details(pod):
    """Parse RunPod pod dict to extract IP and external SSH Port."""
    runtime = pod.get("runtime")
    if not runtime:
        return None, None
        
    ports = runtime.get("ports", [])
    for p in ports:
        # We look for containerPort 22
        if p.get("privatePort") == 22:
            ip = p.get("ip")
            port = p.get("publicPort")
            return ip, port
            
    return None, None

def main():
    zip_path = Path("d:/FYP/CIRCA/CIRCA_runpod.zip")
    if not zip_path.exists():
        logger.error("Required CIRCA_runpod.zip package not found at: %s", zip_path)
        logger.error("Please run package builder script first.")
        sys.exit(1)
        
    gpu_types = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090"]
    deployed_pod = None
    
    # 1. Spawn Pod
    for gpu_type in gpu_types:
        logger.info("Attempting to deploy pod on %s...", gpu_type)
        try:
            pod = runpod.create_pod(
                name="circa-yolov12-training",
                image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                gpu_type_id=gpu_type,
                cloud_type="COMMUNITY",  # Use cheaper community instances
                gpu_count=1,
                volume_in_gb=40,
                container_disk_in_gb=20,
                ports="22/tcp,8888/http",
                start_ssh=True
            )
            deployed_pod = pod
            logger.info("Pod deployment mutation successful. Pod ID: %s", pod["id"])
            break
        except Exception as e:
            logger.warning("Failed to deploy on %s: %s", gpu_type, e)
            
    if not deployed_pod:
        logger.error("Could not deploy pod on any of the target GPUs (no availability or quota error).")
        sys.exit(1)
        
    pod_id = deployed_pod["id"]
    
    # 2. Wait for RUNNING status and IP/Port info
    logger.info("Waiting for pod %s to boot and initialize...", pod_id)
    ip, port = None, None
    for attempt in range(60): # 10 minutes max
        try:
            pod_status = runpod.get_pod(pod_id)
            status = pod_status.get("desiredStatus")
            
            # Print status update
            logger.info("  Attempt %d: Status is '%s'...", attempt + 1, status)
            
            if status == "RUNNING" or pod_status.get("runtime"):
                ip, port = get_ssh_connection_details(pod_status)
                if ip and port:
                    logger.info("Pod is running! SSH connection details: root@%s:%d", ip, port)
                    break
        except Exception as e:
            logger.warning("Error fetching status: %s", e)
            
        time.sleep(10)
        
    if not ip or not port:
        logger.error("Failed to acquire SSH connection details for the pod.")
        logger.error("You can terminate the pod manually using: runpod.terminate_pod('%s')", pod_id)
        sys.exit(1)
        
    # Give the SSH daemon another 10 seconds to fully start up inside the container
    logger.info("Sleeping 10s to let SSH daemon start up...")
    time.sleep(10)
    
    # 3. SFTP Upload zip package
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    private_key_path = os.path.expanduser("~/.ssh/id_rsa")
    
    logger.info("Connecting via SSH (using root@%s:%d)...", ip, port)
    connected = False
    for attempt in range(5):
        try:
            pkey = paramiko.RSAKey.from_private_key_file(private_key_path)
            ssh_client.connect(ip, port=port, username="root", pkey=pkey, timeout=15)
            connected = True
            break
        except Exception as e:
            logger.warning("SSH connection attempt %d failed: %s. Retrying in 10s...", attempt + 1, e)
            time.sleep(10)
            
    if not connected:
        logger.error("Could not establish SSH connection to the pod.")
        sys.exit(1)
        
    logger.info("Uploading CIRCA_runpod.zip (1.73 GB) via SFTP. Please wait...")
    try:
        sftp = ssh_client.open_sftp()
        # Define callback to show progress percentage
        last_pct = [0]
        def upload_progress(transferred, total):
            pct = int((transferred / total) * 100)
            if pct % 10 == 0 and pct != last_pct[0]:
                logger.info("  Upload progress: %d%% completed...", pct)
                last_pct[0] = pct
                
        sftp.put(str(zip_path), "/workspace/CIRCA_runpod.zip", callback=upload_progress)
        sftp.close()
        logger.info("SFTP Upload completed successfully.")
    except Exception as e:
        logger.error("SFTP Upload failed: %s", e)
        ssh_client.close()
        sys.exit(1)
        
    # 4. Trigger Remote Setup and Training via SSH
    logger.info("Starting remote environment setup and training scripts...")
    
    # We construct a remote command that runs setup, then triggers HPO/Fine-tuning in the background
    remote_command = (
        "cd /workspace && "
        "python3 -m zipfile -e CIRCA_runpod.zip CIRCA && "
        "cd CIRCA && "
        "bash scripts/core/setup-runpod-environment.sh > setup_runpod.log 2>&1 && "
        "nohup bash -c '"
        "python3 train_engine.py --mode train --variant s --epochs 50 --id 008 --desc copypaste_small --data datasets/unified_pcb_v3_copypaste/data.yaml > train_engine_s.log 2>&1 && "
        "python3 train_engine.py --mode train --variant m --epochs 50 --id 009 --desc copypaste_medium --data datasets/unified_pcb_v3_copypaste/data.yaml > train_engine_m.log 2>&1"
        "' > train_execution.log 2>&1 &"
    )
    
    try:
        stdin, stdout, stderr = ssh_client.exec_command(remote_command)
        # We wait briefly to ensure command was sent successfully
        time.sleep(5)
        logger.info("Training initiated successfully in background!")
        logger.info("Monitor logs remotely via SSH:")
        logger.info("  Small:  ssh -i %s -p %d root@%s 'tail -f /workspace/CIRCA/train_engine_s.log'", private_key_path, port, ip)
        logger.info("  Medium: ssh -i %s -p %d root@%s 'tail -f /workspace/CIRCA/train_engine_m.log'", private_key_path, port, ip)
    except Exception as e:
        logger.error("Failed to run commands via SSH: %s", e)
    finally:
        ssh_client.close()

if __name__ == "__main__":
    main()
