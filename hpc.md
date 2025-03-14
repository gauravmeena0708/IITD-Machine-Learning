qsub -I -P ail721.aib242286.course -l select=1:ncpus=4:ngpus=1:mem=24G:centos=skylake -l walltime=01:30:00  

qstat -T -u aib242286

wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh  
bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh
