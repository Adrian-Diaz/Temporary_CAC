cd ../src/
if [ "$1" == "-p" ]; then
  make mpi 
  cp lmp_mpi ../inputs/
elif [ "$1" == "-s" ]; then
  make serial
  cp lmp_serial ../inputs/
else
  echo "-p or -s"
fi
