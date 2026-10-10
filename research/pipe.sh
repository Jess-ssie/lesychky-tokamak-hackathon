#!/bin/bash
# відтворення набору команди з тих самих seed + підготовка ознак
set -u
cd /home/claude/work
G=/tmp/claude-0/g; T=/home/claude/toksym_zip/participant/toksym-linux
for spec in "102 1000" "105 2000" "106 2000"; do
  set -- $spec; S=$1; N=$2
  echo "[$(date +%H:%M:%S)] gen seed $S x $N"
  $T gen --shots $N --seed $S --out $G/s$S > $G/s$S.log 2>&1
  cmp $G/s$S/shots.csv team_shots_$S.csv && echo "  shots.csv seed $S: IDENTICAL to team file" || echo "  shots.csv seed $S: DIFFERS"
  head -2 $G/s$S/measured.csv | tail -1 | cut -c1-80
  python3 -I prep2.py $G/s$S/measured.csv s$S.pkl && cp $G/s$S/shots.csv shots_s$S.csv && rm -f $G/s$S/measured.csv
  python3 -I big.py s$S.pkl shots_s$S.csv s${S}F.pkl
  echo "[$(date +%H:%M:%S)] done seed $S"
done
echo ALLDONE
