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


running on GPU-Cluster **not working yet**
srun --ntasks=1 --cpus-per-task=4 --mem=12G --gres=gpu:1 singularity exec ./w2v_did.simg --gpus all -e "TEST=../data/reisedom/data/test_segmented/" -e "TRAIN=../data/reisedom/data/train_segmented/" -e "MODEL=../data/reisedom/data/models/xlsr_53_56k.pt" -e "EPOCHS=10" -e "BSIZE=15" -v ../data/reisedom/data:/data fiviapas/w2v_did