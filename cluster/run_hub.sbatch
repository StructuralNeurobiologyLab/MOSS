#!/bin/bash
#SBATCH --job-name=moss_hub
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/lmb/home/nmedina/MOSS_hub/hub_%j.out

PROJECTS_DIR=/ceph.groups/jkornfeld.grp/nmedina/moss_cluster/hub_projects
mkdir -p "$PROJECTS_DIR"
NODE_IP=$(hostname -I | awk '{print $1}')
echo "======================================================================"
echo "MOSS HUB  node=$(hostname)  IP=$NODE_IP  $(date)"
echo "  web console : http://$NODE_IP:8080/"
echo "  clients     : $NODE_IP:8765   (MOSS -> Multi-User -> Join -> LAN)"
echo "  projects_dir: $PROJECTS_DIR"
echo "======================================================================"
cd /lmb/home/nmedina/MOSS_hub
export PYTHONPATH=$PWD
export QT_QPA_PLATFORM=offscreen
exec /lmb/home/nmedina/miniconda3/envs/moss/bin/python -u -m segmentation_suite.hub \
  --host "$NODE_IP" --port 8765 --web-port 8080 \
  --projects-dir "$PROJECTS_DIR"
