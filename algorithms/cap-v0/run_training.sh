# use ps -ef to look at all running processes
# use kill proc_number to stop the process

for ((i=1; i<=4; i++))
do
    python dqn.py $i &
    echo "Trial $i Started"
done
