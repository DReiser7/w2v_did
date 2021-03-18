# enter into script folder with git bash and run over " start merge-docker-fork.sh "

cd ../
cd ../

if [ -d "/f_w2v_did" ]
then
   mkdir f_w2v_did
   cd f_w2v_did
   git clone https://github.com/Synoon/w2v_did.git
   git remote add upstream https://github.com/DReiser7/w2v_did.git
fi


cd f_w2v_did/w2v_did
git fetch upstream
git merge upstream/master
git push

read  -n 1 -p "Input Selection:" mainmenuinput