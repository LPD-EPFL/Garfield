pwd=`pwd`
while read p; do
        ssh ${p%:*} "pkill -f $pwd/rpc_bench.py" < /dev/tty
done < "nodes"
