#include "libhipeless.h"
#include "minunit.h"

int tests_run = 0;

static const char* test_minunit_sanity() {
  mu_assert("error, 1 != 1", 1 == 1);
  return 0;
}

static const char* all_tests() {
  mu_run_test(test_minunit_sanity);
  return 0;
}

int main(int argc, char* argv[]) {
  const char* result = all_tests();
  if(result != 0) {
    printf("%s\n", result);
  }
  else {
    printf("ALL TESTS PASSED\n");
  }
  printf("Tests run: %d\n", tests_run);

  return result != 0;
}
