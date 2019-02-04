#!/bin/bash

# Script to check policy_perf_test code works with each possible combo of options

echo "Performance test results for parallel_reduce code computing sum of sequence [0,N) with various (nested) policies"

EXECUTABLE=policy_performance

TEAMRANGE=1000
THREADRANGE=4
VECTORRANGE=32
TEAMSIZE=4
VECTORSIZE=1
OREPEAT=1
MREPEAT=1
IREPEAT=1
SCHEDULE=1

SUFFIX=host
if [ -e $EXECUTABLE.$SUFFIX ]
then
SCHEDULE=1
echo "Host tests Static schedule"
for CODE in {100,110,111,112,120,121,122,200,210,211,212,220,221,222,300,400,500}
do
  OMP_PROC_BIND=true ./$EXECUTABLE.$SUFFIX $TEAMRANGE $THREADRANGE $VECTORRANGE $OREPEAT $MREPEAT $IREPEAT $TEAMSIZE $VECTORSIZE $SCHEDULE $CODE
done

SCHEDULE=2
echo "Host tests Dynamic schedule"
for CODE in {100,110,111,112,120,121,122,200,210,211,212,220,221,222,300,400,500}
do
  OMP_PROC_BIND=true ./$EXECUTABLE.$SUFFIX $TEAMRANGE $THREADRANGE $VECTORRANGE $OREPEAT $MREPEAT $IREPEAT $TEAMSIZE $VECTORSIZE $SCHEDULE $CODE
done
fi

SUFFIX=cuda
if [ -e $EXECUTABLE.$SUFFIX ]
then
SCHEDULE=1
echo "Cuda tests Static schedule"
for CODE in {100,110,111,112,120,121,122,200,210,211,212,220,221,222,300,400,500}
do
  ./$EXECUTABLE.$SUFFIX $TEAMRANGE $THREADRANGE $VECTORRANGE $OREPEAT $MREPEAT $IREPEAT $TEAMSIZE $VECTORSIZE $SCHEDULE $CODE
done

SCHEDULE=2
echo "Cuda tests Dynamic schedule"
for CODE in {100,110,111,112,120,121,122,200,210,211,212,220,221,222,300,400,500}
do
  ./$EXECUTABLE.$SUFFIX $TEAMRANGE $THREADRANGE $VECTORRANGE $OREPEAT $MREPEAT $IREPEAT $TEAMSIZE $VECTORSIZE $SCHEDULE $CODE
done
fi
