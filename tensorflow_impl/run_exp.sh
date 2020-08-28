ps_names="servers"
worker_names="workers"
pservers=""
workers=""
max_steps=102
proto=vanilla
exp=mnist
batch=250
byzwk=0
byzps=0
num_wrk=100					#maximum number of workers
num_serv=$(( 2*$byzps + 3 ))
echo "Number of working servers: $num_serv"
eval_steps=20
pwd=`pwd`
i=0
while read p; do
	pservers=$pservers",$p";
    	i=$((i+1))
	if [ $i -eq $num_serv ]
  	then
     		break
  	fi
done < $ps_names
pservers=${pservers:1} # remove first ','
i=0
while read p; do
	workers=$workers",$p";
	i=$((i+1))
        if [ $i -eq $num_wrk ]
        then
                break
        fi
done < $worker_names
workers=${workers:1} # remove first ','

line="--nbbyzps $byzps --nbbyzwrk $byzwk --ps_hosts $pservers --worker_hosts $workers --max_steps $max_steps --batch $batch --eval_steps $eval_steps --$proto True --experiment $exp" # --asyncr --smart --log --time_save --less_grad --vanilla

#i=$((i-1))
i=0
while read p; do
	cmd=" python3 $pwd/byzPS.py $line --task_index $i --job_name ps" 
	ssh ${p%:*} "$cmd" < /dev/tty &
	i=$((i+1))
        if [ $i -eq $num_serv ]
        then
                break
        fi
done < $ps_names

sleep 10 				#Give PSes some time to connect to each other....then workers can start, connect, and communicate
i=0
while read p; do
	cmd=" python3 $pwd/byzWorker.py $line --task_index $i --job_name worker" 
	ssh ${p%:*} "$cmd" < /dev/tty &
	i=$((i+1))
        if [ $i -eq $num_wrk ]
        then
                break
        fi
done < $worker_names
