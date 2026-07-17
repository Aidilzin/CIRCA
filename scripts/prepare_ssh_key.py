import os
import subprocess
import runpod
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SSHPrep")

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

def main():
    ssh_dir = Path(os.path.expanduser("~/.ssh"))
    ssh_dir.mkdir(exist_ok=True)
    
    private_key_path = ssh_dir / "id_rsa"
    public_key_path = ssh_dir / "id_rsa.pub"
    
    if not private_key_path.exists():
        logger.info("Generating passwordless SSH key pair...")
        try:
            subprocess.run([
                "ssh-keygen", "-t", "rsa", "-b", "4096",
                "-f", str(private_key_path), "-N", ""
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Successfully generated SSH key pair.")
        except Exception as e:
            logger.error("Failed to generate SSH key pair: %s", e)
            return False
    else:
        logger.info("Existing SSH key pair found.")
        
    # Read public key
    with open(public_key_path, "r") as f:
        pub_key = f.read().strip()
        
    logger.info("Uploading public key to RunPod account...")
    try:
        user = runpod.get_user()
        current_pub_key = user.get("pubKey")
        if current_pub_key != pub_key:
            runpod.update_user_settings(pubkey=pub_key)
            logger.info("Successfully updated public key on RunPod!")
        else:
            logger.info("Public key on RunPod is already up to date.")
        return True
    except Exception as e:
        logger.error("Failed to upload SSH key to RunPod: %s", e)
        return False

if __name__ == "__main__":
    main()
