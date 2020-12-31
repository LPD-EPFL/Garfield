pwd=`pwd`
while read p; do
	ssh ${p%:*} "pkill -f $pwd/trainer.py" < /dev/tty
done < "workers"
while read p; do
        ssh ${p%:*} "pkill -f $pwd/trainer.py" < /dev/tty
done < "servers"
