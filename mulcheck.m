load out;
isequalRel = @(x,y,tol) (all(all(( abs(x-y) <= ( tol*max(abs(x),abs(y)) + eps))) ));
if isequalRel(A*B, C, 1e-5);
  printf("Correct\n");
else
  printf("Incorrect\n");
endif
