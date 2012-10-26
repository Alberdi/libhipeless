load out;
if all(all(abs(A*B-C) < 0.00001));
  printf("Correct\n");
else
  printf("Incorrect\n");
endif
