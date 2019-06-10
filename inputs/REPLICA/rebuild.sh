cd ../../src/
if [ "$1" == "-p" ]; then
  make yes-replica
  make mpi 
  cp -v lmp_mpi ../inputs/REPLICA
elif [ "$1" == "-s" ]; then
  make yes-replica
  make serial
  cp -v lmp_serial ../inputs/REPLICA
else
  echo "-p or -s"
fi