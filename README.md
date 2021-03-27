# Docker
## local build
`docker build -t w2v_did`

## CI/CD hub.docker
Starting Docker hub pipeline
'git push ' local changes
in shell
```bash
cd <dir>/w2v_did
start scripts/merge-docker-fork.sh
```
Built will start automatically on https://hub.docker.com/repository/docker/fiviapas/w2v_did/builds

## Pulling docker hub img
`docker pull fiviapas/w2v_did`

running locally
`docker run -d -e "TEST=/data/dev/segmented/" -e "TRAIN=/data/dev/segmented/" -e "MODEL=/data/models/wav2vec_small.pt" -e "EPOCHS=10" -v //c/workarea/w2v_did/data:/data  fiviapas/w2v_did`

running on GPULAND:
`docker run -d --gpus all -e "TEST=/data/test_segmented/" -e "TRAIN=/data/train_segmented/" -e "MODEL=./data/models/xlsr_53_56k.pt" -e "EPOCHS=10" -e "BSIZE=15" -v "$(pwd)"/data:/data fiviapas/w2v_did`


### running on GPU-Cluster
pull image:
```bash
srun --ntasks=1 --cpus-per-task=4 --mem=8G  singularity pull docker://reisedom/w2v_did_wandb
```
create screen session:
```bash
screen -S username-session
```
run image:
```bash
srun --pty --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:1 singularity shell w2v_did_wandb.simg
```
clone git repository:
```bash
git clone https://github.com/DReiser7/w2v_did.git
cd w2v_did
git checkout wandb
cd ..
```
run main application:
```bash
#reisedom
python ./w2v_did/DidMain.py  "/cluster/home/reisedom/data/train_segmented/" "/cluster/home/reisedom/data/test_segmented/" "/cluster/home/reisedom/data/models/xlsr_53_56k.pt" 3 2
#fiviapas
python ./w2v_did/DidMain.py  "/cluster/home/fiviapas/data/train_segmented/" "/cluster/home/fiviapas/data/test_segmented/" "/cluster/home/fiviapas/data/models/xlsr_53_56k.pt" 3 2
```
detach screen session:
```bash
Ctrl+A, D
```
reattach screen session:
```bash
screen -r username-session
```