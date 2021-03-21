# enter into script folder with git bash and run over " start merge-docker-fork.sh "

#cd ../
#cd ../
#
#if [ -d "/f_w2v_did" ]
#then
#   mkdir f_w2v_did
#   cd f_w2v_did
#   git clone https://github.com/Synoon/w2v_did.git
#   cd w2v_did
#   git remote add upstream https://github.com/DReiser7/w2v_did.git
#fi


cd ../f_w2v_did/w2v_did
git fetch upstream
git merge upstream/master
git push

echo "build local img       [ docker build -t w2v_did ]"
echo "pull docker img over  [ docker pull fiviapas/w2v_did ]"
echo "run img over          [ docker run -d fiviapas/w2v_did ]"
echo 'run local img over    [ docker run -d -e "TEST=/data/dev/segmented/" -e "TRAIN=/data/dev/segmented/" -e "MODEL=/data/models/wav2vec_small.pt"  -v //c/workarea/w2v_did/data:/data  fiviapas/w2v_did ]'
echo 'run on GPULAND        [ docker run -d -e "TEST=/data/test_segmented/" -e "TRAIN=/data/train_segmented/" -e "MODEL=/data/models/xlsr_53_56k.pt"  -v "$(pwd)"/data:/data  fiviapas/w2v_did ]'
read  -n 1 -p "" mainmenuinput