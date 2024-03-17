#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void debug(const char *message);

void error(const char *message);

void info(const char *message);

void init_logger_debug(void);

void init_logger_info(void);

void init_logger_trace(void);

void run_all(void);

void rust_lib_helloworld(void);

void trace(const char *message);

void warn(const char *message);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
