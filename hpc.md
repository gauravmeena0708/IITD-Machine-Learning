Simple job

    qsub -I -P ail721.aib242286.course -l select=1:ncpus=4:ngpus=1:mem=24G:centos=skylake -l walltime=01:30:00  
Time approx job list

    qstat -T -u aib242286
    
Adding all terminal output to log file

    python run3.py > log3_run4.txt 2>&1
    
Anaconda Install

    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh  
    bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh  

High Priority q

    qsub -I -P ail721.aib242286.course -q high -l select=1:ncpus=4:ngpus=1:mem=24G:centos=skylake -l walltime=05:00:00  
SCAI q

    qsub -I -P ail721.aib242286.course -q scai_q -l select=1:ncpus=4:ngpus=1:mem=24G:centos=skylake -l walltime=01:30:00 

## Non-interactive job 
    qsub -P scai -q scai_q -l select=1:ncpus=1:ngpus=1:mem=24G:centos=skylake  -l walltime=02:30:00      -o out_train_tabsyn.txt      ~/job2.sh

    qsub -P ail721.aib242286.course      -l select=1:ncpus=4:ngpus=1:mem=24G:centos=skylake      -l walltime=01:30:00      -o out.txt      ~/job.sh

## Tmux

        tmux list

Start a new session or attach to an existing session named mysession  

        tmux new-session -A -s mysession 

kill session  

        tmux kill-session -t mysession

Detach : CTRL+B and d

attach

    tmux attach-session -t mysession

shift sessions

    CTRL+B ( or )
New window on side

    CTRL+B % 
New window on bottom

    CTRL+B " 
Shift between windows

    CTRL+B <- or ->
1. https://tmuxcheatsheet.com/
2. https://gist.github.com/MohamedAlaa/2961058
3. https://www.youtube.com/watch?v=Yl7NFenTgIo
