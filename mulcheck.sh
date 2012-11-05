while true; do
  ./main > out
  octave --silent mulcheck.m
done
