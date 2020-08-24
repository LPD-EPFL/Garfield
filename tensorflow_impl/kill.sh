pwd=`pwd`
while read p; do
	ssh ${p%:*} "pkill -f $pwd/byzWorker.py" < /dev/tty
done < "workers"
while read p; do
        ssh ${p%:*} "pkill -f $pwd/byzPS.py" < /dev/tty
done < "servers"
