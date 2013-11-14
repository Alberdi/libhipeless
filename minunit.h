#define mu_assert(message, test) do { if (!(test)) return message; } while (0)
#define mu_run_test(test) do { const char *message = test(); \
                               if (message) return message; \
                               else printf("Ran test %i.\n", ++tests_run); } while (0)
extern int tests_run;
