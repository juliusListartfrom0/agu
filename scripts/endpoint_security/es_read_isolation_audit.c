// es_read_isolation_audit.c — Endpoint Security read-isolation audit client
//
// TASK-0258 Amendment-001: the verification worker's read-isolation attestation
// requires an "authenticated kernel-audit provider". On macOS this is the
// Endpoint Security framework (es_* C API). This client subscribes to the
// kernel's path-resolution notify events for the audited worker process tree and
// streams a bounded, canonical transcript (JSONL) that ends with a clean
// finalization record after observing the audited target's EXIT event. The Python side can validate the diagnostic projection,
// but still cannot fold it into `agu.task0258-module-a-worker-read-isolation-attestation.v1`
// without a future externally authenticated provider.
//
// BUILD (unsigned; signing + entitlement are a separate platform step):
//   clang -Wall -Wextra -Werror -O2 -framework EndpointSecurity -framework CoreFoundation -lbsm \
//         -o es_read_isolation_audit es_read_isolation_audit.c
//
// LOADING requires a binary signed with the
// `com.apple.developer.endpoint-security.client` entitlement and user approval
// in System Settings -> Privacy & Security -> Endpoint Security.
//
// Usage: es_read_isolation_audit --pid <worker-pid> --out <absolute-transcript.jsonl> [--timeout-seconds <n>]
// The diagnostic transcript retains notify auth/flags results and the
// available per-event/global sequence numbers. Exit 3 means a size/lineage
// overflow, 4 means the bounded timeout fired, 5 means a protocol or
// sequence-gap failure, and 6 means the target EXIT was not observed or the
// observation was interrupted. None of these exit states is an attestation.

#include <EndpointSecurity/EndpointSecurity.h>
#include <bsm/libbsm.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_ROWS 21600
#define MAX_ROW_BYTES 512
#define MAX_TRANSCRIPT_BYTES 16777216ULL
#define MAX_TRACKED_PROCESSES 1024

static volatile sig_atomic_t g_stop = 0;
static int g_target_pid = -1;
static FILE *g_out = NULL;
static uint64_t g_row_count = 0;
static uint64_t g_row_bytes = 0;
static int g_overflow = 0;
static int g_timed_out = 0;
static int g_interrupted = 0;
static int g_target_exit_observed = 0;
static unsigned g_timeout_seconds = 120;
static int g_target_pidversion = -1;
static int g_sequence_gap = 0;
static int g_protocol_error = 0;
static uint64_t g_last_global_seq_num = 0;
static int g_have_global_seq_num = 0;
static uint64_t g_last_seq_num[ES_EVENT_TYPE_LAST];
static uint8_t g_have_seq_num[ES_EVENT_TYPE_LAST];

static void on_signal(int sig) { (void)sig; g_interrupted = 1; g_stop = 1; }
static void on_timeout(int sig) { (void)sig; g_timed_out = 1; g_stop = 1; }

typedef struct {
    pid_t pid;
    int pidversion;
} tracked_process_t;

static tracked_process_t g_tracked_processes[MAX_TRACKED_PROCESSES];
static size_t g_tracked_process_count = 0;

// Minimal safe event-name for the projection.
static const char *event_name(es_event_type_t t) {
    switch (t) {
        case ES_EVENT_TYPE_NOTIFY_OPEN: return "open";
        case ES_EVENT_TYPE_NOTIFY_STAT: return "stat";
        case ES_EVENT_TYPE_NOTIFY_ACCESS: return "access";
        case ES_EVENT_TYPE_NOTIFY_READLINK: return "readlink";
        case ES_EVENT_TYPE_NOTIFY_READDIR: return "getdents";
        case ES_EVENT_TYPE_NOTIFY_EXEC: return "exec";
        case ES_EVENT_TYPE_NOTIFY_MMAP: return "mmap";
        case ES_EVENT_TYPE_NOTIFY_GETEXTATTR: return "getxattr";
        case ES_EVENT_TYPE_NOTIFY_SETEXTATTR: return "setxattr";
        default: return "other";
    }
}

// Resolve the audited path for a message (open/stat/access/readlink/exec use
// `path`; readdir uses `dir`; xattr uses `attr.name`-style path resolution).
static const es_file_t *message_file(const es_message_t *msg) {
    switch (msg->event_type) {
        case ES_EVENT_TYPE_NOTIFY_OPEN:
            return msg->event.open.file;
        case ES_EVENT_TYPE_NOTIFY_STAT:
            return msg->event.stat.target;
        case ES_EVENT_TYPE_NOTIFY_ACCESS:
            return msg->event.access.target;
        case ES_EVENT_TYPE_NOTIFY_READLINK:
            return msg->event.readlink.source;
        case ES_EVENT_TYPE_NOTIFY_READDIR:
            return msg->event.readdir.target;
        case ES_EVENT_TYPE_NOTIFY_EXEC:
            return msg->event.exec.target->executable;
        case ES_EVENT_TYPE_NOTIFY_MMAP:
            return msg->event.mmap.source;
        case ES_EVENT_TYPE_NOTIFY_GETEXTATTR:
            return msg->event.getextattr.target;
        case ES_EVENT_TYPE_NOTIFY_SETEXTATTR:
            return msg->event.setextattr.target;
        default:
            return NULL;
    }
}

static int track_process(audit_token_t token) {
    pid_t pid = audit_token_to_pid(token);
    int pidversion = audit_token_to_pidversion(token);
    for (size_t i = 0; i < g_tracked_process_count; i++) {
        if (g_tracked_processes[i].pid == pid &&
            g_tracked_processes[i].pidversion == pidversion) {
            return 1;
        }
    }
    if (g_tracked_process_count >= MAX_TRACKED_PROCESSES) {
        g_overflow = 1;
        g_stop = 1;
        return 0;
    }
    if (g_tracked_process_count == 0 && pid == g_target_pid) {
        g_target_pidversion = pidversion;
    }
    g_tracked_processes[g_tracked_process_count++] = (tracked_process_t){pid, pidversion};
    return 1;
}

static int process_is_tracked(const es_message_t *msg) {
    const es_process_t *process = msg->process;
    pid_t pid = audit_token_to_pid(process->audit_token);
    int pidversion = audit_token_to_pidversion(process->audit_token);
    for (size_t i = 0; i < g_tracked_process_count; i++) {
        if (g_tracked_processes[i].pid == pid &&
            g_tracked_processes[i].pidversion == pidversion) {
            return 1;
        }
    }
    if (pid == g_target_pid && g_tracked_process_count == 0) {
        return track_process(process->audit_token);
    }
    if (msg->version >= 4) {
        for (size_t i = 0; i < g_tracked_process_count; i++) {
            audit_token_t parent_token = process->parent_audit_token;
            if (g_tracked_processes[i].pid == audit_token_to_pid(parent_token) &&
                g_tracked_processes[i].pidversion == audit_token_to_pidversion(parent_token)) {
                return track_process(process->audit_token);
            }
        }
    }
    return 0;
}

static int append_bytes(char *buffer, size_t capacity, size_t *used, const char *data, size_t length) {
    if (length > capacity - *used) {
        return 0;
    }
    memcpy(buffer + *used, data, length);
    *used += length;
    return 1;
}

static int append_u64_or_null(char *buffer, size_t capacity, size_t *used,
                              uint64_t value, int available) {
    if (!available) {
        return append_bytes(buffer, capacity, used, "null", 4);
    }
    char encoded[32];
    int written = snprintf(encoded, sizeof(encoded), "%llu", (unsigned long long)value);
    return written > 0 && (size_t)written < sizeof(encoded) &&
           append_bytes(buffer, capacity, used, encoded, (size_t)written);
}

static int append_json_escaped(char *buffer, size_t capacity, size_t *used,
                               const uint8_t *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        uint8_t byte = data[i];
        if (byte == '"') {
            if (!append_bytes(buffer, capacity, used, "\\\"", 2)) return 0;
        } else if (byte == '\\') {
            if (!append_bytes(buffer, capacity, used, "\\\\", 2)) return 0;
        } else if (byte == '\b') {
            if (!append_bytes(buffer, capacity, used, "\\b", 2)) return 0;
        } else if (byte == '\f') {
            if (!append_bytes(buffer, capacity, used, "\\f", 2)) return 0;
        } else if (byte == '\n') {
            if (!append_bytes(buffer, capacity, used, "\\n", 2)) return 0;
        } else if (byte == '\r') {
            if (!append_bytes(buffer, capacity, used, "\\r", 2)) return 0;
        } else if (byte == '\t') {
            if (!append_bytes(buffer, capacity, used, "\\t", 2)) return 0;
        } else if (byte < 0x20) {
            char escaped[7];
            int written = snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
            if (written != 6 || !append_bytes(buffer, capacity, used, escaped, (size_t)written)) return 0;
        } else if (byte < 0x80) {
            if (!append_bytes(buffer, capacity, used, (const char *)&byte, 1)) return 0;
        } else {
            size_t sequence_length;
            if (byte >= 0xc2 && byte <= 0xdf) {
                sequence_length = 2;
            } else if (byte >= 0xe0 && byte <= 0xef) {
                sequence_length = 3;
            } else if (byte >= 0xf0 && byte <= 0xf4) {
                sequence_length = 4;
            } else {
                return 0;
            }
            if (i + sequence_length > length) return 0;
            if ((sequence_length >= 2 && (data[i + 1] & 0xc0) != 0x80) ||
                (sequence_length >= 3 && (data[i + 2] & 0xc0) != 0x80) ||
                (sequence_length >= 4 && (data[i + 3] & 0xc0) != 0x80)) {
                return 0;
            }
            if ((byte == 0xe0 && data[i + 1] < 0xa0) ||
                (byte == 0xed && data[i + 1] >= 0xa0) ||
                (byte == 0xf0 && data[i + 1] < 0x90) ||
                (byte == 0xf4 && data[i + 1] > 0x8f) ||
                !append_bytes(buffer, capacity, used, (const char *)&data[i], sequence_length)) {
                return 0;
            }
            i += sequence_length - 1;
        }
    }
    return 1;
}

static int append_notify_result(char *buffer, size_t capacity, size_t *used,
                                const es_message_t *msg) {
    if (msg == NULL || msg->action_type != ES_ACTION_TYPE_NOTIFY) {
        return 0;
    }
    if (msg->action.notify.result_type == ES_RESULT_TYPE_AUTH) {
        const char *auth_result;
        switch (msg->action.notify.result.auth) {
            case ES_AUTH_RESULT_ALLOW:
                auth_result = "allow";
                break;
            case ES_AUTH_RESULT_DENY:
                auth_result = "deny";
                break;
            default:
                return 0;
        }
        return append_bytes(buffer, capacity, used,
                            ",\"result_type\":\"auth\",\"result_auth\":\"",
                            sizeof(",\"result_type\":\"auth\",\"result_auth\":\"") - 1) &&
               append_bytes(buffer, capacity, used, auth_result, strlen(auth_result)) &&
               append_bytes(buffer, capacity, used, "\"", 1);
    }
    if (msg->action.notify.result_type == ES_RESULT_TYPE_FLAGS) {
        char encoded[32];
        int written = snprintf(encoded, sizeof(encoded),
                                ",\"result_type\":\"flags\",\"result_flags\":%u",
                                msg->action.notify.result.flags);
        return written > 0 && (size_t)written < sizeof(encoded) &&
               append_bytes(buffer, capacity, used, encoded, (size_t)written);
    }
    return 0;
}

static int record_message_sequence(const es_message_t *msg) {
    if (msg == NULL || msg->action_type != ES_ACTION_TYPE_NOTIFY) {
        g_protocol_error = 1;
        g_stop = 1;
        return 0;
    }
    if (msg->version >= 2) {
        unsigned event_type = (unsigned)msg->event_type;
        if (event_type >= (unsigned)ES_EVENT_TYPE_LAST) {
            g_protocol_error = 1;
            g_stop = 1;
            return 0;
        }
        if (g_have_seq_num[event_type] &&
            (g_last_seq_num[event_type] == UINT64_MAX ||
             msg->seq_num != g_last_seq_num[event_type] + 1ULL)) {
            g_sequence_gap = 1;
            g_stop = 1;
            return 0;
        }
        g_last_seq_num[event_type] = msg->seq_num;
        g_have_seq_num[event_type] = 1;
    }
    if (msg->version >= 4) {
        if (g_have_global_seq_num &&
            (g_last_global_seq_num == UINT64_MAX ||
             msg->global_seq_num != g_last_global_seq_num + 1ULL)) {
            g_sequence_gap = 1;
            g_stop = 1;
            return 0;
        }
        g_last_global_seq_num = msg->global_seq_num;
        g_have_global_seq_num = 1;
    }
    return 1;
}

static int write_event_row(const es_message_t *msg, const es_process_t *process,
                           const char *name, const es_file_t *file) {
    char row[MAX_ROW_BYTES];
    size_t used = 0;
    pid_t pid = audit_token_to_pid(process->audit_token);
    int pidversion = audit_token_to_pidversion(process->audit_token);
    int written = snprintf(row, sizeof(row),
                           "{\"event\":\"%s\",\"pid\":%d,\"pidversion\":%d,\"ppid\":%d,\"seq_num\":",
                           name, pid, pidversion, process->ppid);
    if (written <= 0 || (size_t)written >= sizeof(row)) {
        return 0;
    }
    used = (size_t)written;
    if (!append_u64_or_null(row, sizeof(row), &used, msg->seq_num, msg->version >= 2) ||
        !append_bytes(row, sizeof(row), &used, ",\"global_seq_num\":",
                      sizeof(",\"global_seq_num\":") - 1) ||
        !append_u64_or_null(row, sizeof(row), &used, msg->global_seq_num, msg->version >= 4) ||
        !append_bytes(row, sizeof(row), &used, ",\"path\":", sizeof(",\"path\":") - 1)) {
        return 0;
    }
    if (file == NULL || file->path.data == NULL) {
        if (!append_bytes(row, sizeof(row), &used, "null", 4)) return 0;
    } else {
        if (file->path_truncated) return 0;
        if (!append_bytes(row, sizeof(row), &used, "\"", 1) ||
            !append_json_escaped(row, sizeof(row), &used, (const uint8_t *)file->path.data, file->path.length) ||
            !append_bytes(row, sizeof(row), &used, "\"", 1)) {
            return 0;
        }
    }
    if (!append_notify_result(row, sizeof(row), &used, msg) ||
        !append_bytes(row, sizeof(row), &used, "}\n", 2)) {
        return 0;
    }
    if (g_row_count >= MAX_ROWS || g_row_bytes + (uint64_t)used > MAX_TRANSCRIPT_BYTES) {
        g_overflow = 1;
        g_stop = 1;
        return 0;
    }
    if (fwrite(row, 1, used, g_out) != used) {
        g_overflow = 1;
        g_stop = 1;
        return 0;
    }
    g_row_count++;
    g_row_bytes += (uint64_t)used;
    return 1;
}

static void handler(es_client_t *client, const es_message_t *msg) {
    (void)client;
    if (g_stop) {
        return;
    }
    if (!record_message_sequence(msg)) {
        return;
    }
    if (msg->event_type == ES_EVENT_TYPE_NOTIFY_EXIT) {
        if (process_is_tracked(msg) &&
            audit_token_to_pid(msg->process->audit_token) == g_target_pid &&
            audit_token_to_pidversion(msg->process->audit_token) == g_target_pidversion) {
            alarm(0);
            g_target_exit_observed = 1;
            g_stop = 1;
        }
        return;
    }
    if (msg->event_type == ES_EVENT_TYPE_NOTIFY_FORK) {
        if (process_is_tracked(msg)) {
            track_process(msg->event.fork.child->audit_token);
            if (!write_event_row(msg, msg->event.fork.child, "fork", NULL)) {
                g_protocol_error = 1;
                g_stop = 1;
            }
        }
        return;
    }
    if (!process_is_tracked(msg)) {
        return;
    }
    if (!write_event_row(msg, msg->process, event_name(msg->event_type), message_file(msg))) {
        g_protocol_error = 1;
        g_stop = 1;
    }
}

static void usage(void) {
    fprintf(stderr, "usage: es_read_isolation_audit --pid <worker-pid> --out <absolute-transcript.jsonl> [--timeout-seconds <n>]\n");
}

static int parse_positive_decimal(const char *text, unsigned long maximum, unsigned long *result) {
    if (text == NULL || *text == '\0') {
        return 0;
    }
    unsigned long parsed = 0;
    for (const unsigned char *cursor = (const unsigned char *)text; *cursor != '\0'; cursor++) {
        if (*cursor < '0' || *cursor > '9') {
            return 0;
        }
        unsigned long digit = (unsigned long)(*cursor - '0');
        if (parsed > (maximum - digit) / 10) {
            return 0;
        }
        parsed = parsed * 10 + digit;
    }
    if (parsed == 0) {
        return 0;
    }
    *result = parsed;
    return 1;
}

static int parse_pid(const char *text, pid_t *result) {
    unsigned long parsed = 0;
    if (!parse_positive_decimal(text, (unsigned long)INT_MAX, &parsed)) {
        return 0;
    }
    *result = (pid_t)parsed;
    return 1;
}

static int parse_timeout(const char *text, unsigned *result) {
    unsigned long parsed = 0;
    if (!parse_positive_decimal(text, (unsigned long)UINT_MAX, &parsed)) {
        return 0;
    }
    *result = (unsigned)parsed;
    return 1;
}

static int open_output_no_follow(const char *path) {
    if (path == NULL || path[0] != '/' || path[1] == '/' || strlen(path) >= PATH_MAX || strstr(path, "//") != NULL) {
        errno = EINVAL;
        return -1;
    }
    char *copy = strdup(path);
    if (copy == NULL) {
        return -1;
    }
    char *last_slash = strrchr(copy, '/');
    if (last_slash == NULL || last_slash[1] == '\0' ||
        strcmp(last_slash + 1, ".") == 0 || strcmp(last_slash + 1, "..") == 0) {
        free(copy);
        errno = EINVAL;
        return -1;
    }
    char *leaf = last_slash + 1;
    int directory_fd = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory_fd < 0) {
        free(copy);
        return -1;
    }
    if (last_slash != copy) {
        *last_slash = '\0';
        char *save = NULL;
        for (char *component = strtok_r(copy + 1, "/", &save);
             component != NULL;
             component = strtok_r(NULL, "/", &save)) {
            if (strcmp(component, ".") == 0 || strcmp(component, "..") == 0) {
                close(directory_fd);
                free(copy);
                errno = EINVAL;
                return -1;
            }
            int next_fd = openat(directory_fd, component,
                                 O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
            if (next_fd < 0) {
                close(directory_fd);
                free(copy);
                return -1;
            }
            close(directory_fd);
            directory_fd = next_fd;
        }
    }
    int output_fd = openat(directory_fd, leaf,
                           O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    int saved_errno = errno;
    close(directory_fd);
    free(copy);
    errno = saved_errno;
    return output_fd;
}

static int write_final_row(void) {
    char row[MAX_ROW_BYTES];
    int written = snprintf(
        row, sizeof(row),
        "{\"record_type\":\"final\",\"rows\":%llu,\"bytes\":%llu,"
        "\"overflow\":%s,\"sequence_gap\":%s,\"protocol_error\":%s,\"timed_out\":%s,"
        "\"target_exit_observed\":%s,\"interrupted\":%s}\n",
        (unsigned long long)g_row_count,
        (unsigned long long)g_row_bytes,
        g_overflow ? "true" : "false",
        g_sequence_gap ? "true" : "false",
        g_protocol_error ? "true" : "false",
        g_timed_out ? "true" : "false",
        g_target_exit_observed ? "true" : "false",
        g_interrupted ? "true" : "false");
    if (written <= 0 || (size_t)written >= sizeof(row)) {
        return 0;
    }
    if (g_row_bytes + (uint64_t)written > MAX_TRANSCRIPT_BYTES) {
        return 0;
    }
    return fwrite(row, 1, (size_t)written, g_out) == (size_t)written;
}

int main(int argc, char **argv) {
    const char *out_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pid") == 0 && i + 1 < argc) {
            if (!parse_pid(argv[++i], &g_target_pid)) {
                usage();
                return 2;
            }
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--timeout-seconds") == 0 && i + 1 < argc) {
            if (!parse_timeout(argv[++i], &g_timeout_seconds)) {
                usage();
                return 2;
            }
        } else {
            usage();
            return 2;
        }
    }
    if (g_target_pid <= 0 || out_path == NULL) {
        usage();
        return 2;
    }
    int output_fd = open_output_no_follow(out_path);
    if (output_fd < 0) {
        fprintf(stderr, "cannot open transcript: %s\n", out_path);
        return 1;
    }
    g_out = fdopen(output_fd, "w");
    if (g_out == NULL) {
        close(output_fd);
        return 1;
    }
    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);
    signal(SIGALRM, on_timeout);

    es_client_t *client = NULL;
        es_new_client_result_t rc = es_new_client(
        &client,
        ^(es_client_t *_Nonnull client, const es_message_t *_Nonnull msg) {
            handler(client, msg);
        });
    if (rc != ES_NEW_CLIENT_RESULT_SUCCESS) {
        fprintf(stderr, "es_new_client failed: %d\n", rc);
        fclose(g_out);
        return 1;
    }
    const es_event_type_t subscribed[] = {
        ES_EVENT_TYPE_NOTIFY_OPEN,
        ES_EVENT_TYPE_NOTIFY_STAT,
        ES_EVENT_TYPE_NOTIFY_ACCESS,
        ES_EVENT_TYPE_NOTIFY_READLINK,
        ES_EVENT_TYPE_NOTIFY_READDIR,
        ES_EVENT_TYPE_NOTIFY_EXEC,
        ES_EVENT_TYPE_NOTIFY_MMAP,
        ES_EVENT_TYPE_NOTIFY_GETEXTATTR,
        ES_EVENT_TYPE_NOTIFY_SETEXTATTR,
        ES_EVENT_TYPE_NOTIFY_FORK,
        ES_EVENT_TYPE_NOTIFY_EXIT,
    };
    es_return_t sub = es_subscribe(client, subscribed,
                                   sizeof(subscribed) / sizeof(subscribed[0]));
    if (sub != ES_RETURN_SUCCESS) {
        fprintf(stderr, "es_subscribe failed: %d\n", sub);
        es_delete_client(client);
        fclose(g_out);
        return 1;
    }
    alarm(g_timeout_seconds);
    while (!g_stop) {
        pause();
    }
    alarm(0);
    es_delete_client(client);
    int final_row_result = write_final_row();
    int flush_result = final_row_result == 1 ? fflush(g_out) : -1;
    int sync_result = flush_result == 0 ? fsync(fileno(g_out)) : -1;
    int close_result = fclose(g_out);
    if (flush_result != 0 || sync_result != 0 || close_result != 0) {
        fprintf(stderr, "transcript finalize failed\n");
        return 1;
    }
    fprintf(stderr, "rows=%llu bytes=%llu overflow=%d sequence_gap=%d protocol_error=%d\n",
            (unsigned long long)g_row_count, (unsigned long long)g_row_bytes,
            g_overflow, g_sequence_gap, g_protocol_error);
    if (g_overflow) {
        return 3;
    }
    if (g_sequence_gap || g_protocol_error) {
        return 5;
    }
    if (g_timed_out) {
        return 4;
    }
    if (g_interrupted || !g_target_exit_observed) {
        return 6;
    }
    return 0;
}
