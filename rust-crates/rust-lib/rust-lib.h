#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


#define NUM_TRAVERSAL 8

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

const size_t *const *build_traversal_seqence(const char *name);

void debug(const char *message);

void error(const char *message);

void info(const char *message);

void init_logger_debug(void);

void init_logger_info(void);

void init_logger_trace(void);

void release_traversal_sequence(const size_t *const *ptr);

void rust_lib_helloworld(void);

void trace(const char *message);

void warn(const char *message);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
