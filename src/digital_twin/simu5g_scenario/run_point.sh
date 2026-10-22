#!/bin/bash
# Runs one point of the load sweep inside the opp_env session. Usage: run_point.sh <send_interval_seconds> <sim_time_s> <repetition>
# (opp_env run simu5g-1.7.0 -c "bash run_point.sh 0.002 6 0" from the workspace directory)
INTERVAL=$1
SIMTIME=$2
REP=$3
cd "$HOME/6g-sim-workspace/simu5g-1.7.0/simulations/nr/standalone" || exit 1
cp "/mnt/c/Users/keert/Downloads/fy/6G/src/digital_twin/simu5g_scenario/loadsweep.ini" . 2>/dev/null
rm -rf "results/LoadSweep/0.vec" "results/LoadSweep/0.sca"
../../../bin/simu5g -u Cmdenv -f loadsweep.ini -c LoadSweep -r 0 "--seed-set=${REP}" --sim-time-limit="${SIMTIME}s" \
  "--*.server.app[*].sendInterval=${INTERVAL}s" > "/tmp/loadsweep_${REP}.log" 2>&1
tail -3 "/tmp/loadsweep_${REP}.log"
ls -la results/LoadSweep 2>&1 | tail -3
python3 /mnt/c/Users/keert/Downloads/fy/6G/src/digital_twin/simu5g_scenario/summarise_vec.py "results/LoadSweep/0.vec" "$4" 1000 "$SIMTIME" 1.0 > "/tmp/loadsweep_${REP}.json"
cat "/tmp/loadsweep_${REP}.json"
