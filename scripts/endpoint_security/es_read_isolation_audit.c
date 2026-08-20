// es_read_isolation_audit.c — Endpoint Security read-isolation audit client
//
// TASK-0258 Amendment-001: the verification worker's read-isolation attestation
// requires an "authenticated kernel-audit provider". On macOS this is the
// Endpoint Security framework (es_* C API). This client subscribes to the
// kernel's path-resolution notify events for the audited worker process tree and
// streams a bounded, canonical transcript (JSONL) that the Python side folds
// into `agu.task0258-module-a-worker-read-isolation-attestation.v1`.
//
// BUILD (unsigned; signing + entitlement are a separate platform step):
//   clang -O2 -framework EndpointSecurity -framework CoreFoundation -lbsm \
//         -o es_read_isolation_audit es_read_isolation_audit.c
//
// LOADING requires a binary signed with the
// `com.apple.developer.endpoint-security.client` entitlement and user approval
// in System Settings -> Privacy & Security -> Endpoint Security.
//
// Usage: es_read_isolation_audit --pid <worker-pid> --out <transcript.jsonl>

#include <EndpointSecurity/EndpointSecurity.h>
#include <bsm/libbsm.h>
#include <errno.h>
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

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) { (void)sig; g_stop = 1; }

static int g_target_pid = -1;
static FILE *g_out = NULL;
static uint64_t g_row_count = 0;
static uint64_t g_row_bytes = 0;
static int g_overflow = 0;

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
static const char *message_path(const es_message_t *msg) {
    switch (msg->event_type) {
        case ES_EVENT_TYPE_NOTIFY_OPEN:
            return msg->event.open.file->path.data != NULL
                       ? strndup((const char *)msg->event.open.file->path.data,
                                 msg->event.open.file->path.length)
                       : NULL;
        case ES_EVENT_TYPE_NOTIFY_STAT:
            return msg->event.stat.target->path.data != NULL
                       ? strndup((const char *)msg->event.stat.target->path.data,
                                 msg->event.stat.target->path.length)
                       : NULL;
        case ES_EVENT_TYPE_NOTIFY_ACCESS:
            return msg->event.access.target->path.data != NULL
                       ? strndup((const char *)msg->event.access.target->path.data,
                                 msg->event.access.target->path.length)
                       : NULL;
        case ES_EVENT_TYPE_NOTIFY_READLINK:
            return msg->event.readlink.source->path.data != NULL
                       ? strndup((const char *)msg->event.readlink.source->path.data,
                                 msg->event.readlink.source->path.length)
                       : NULL;
        case ES_EVENT_TYPE_NOTIFY_READDIR:
            // es_event_readdir_t carries no per-event path in this SDK; the
            // directory is identified by the audited process context only.
            return NULL;
        case ES_EVENT_TYPE_NOTIFY_EXEC:
            return msg->event.exec.target->executable->path.data != NULL
                       ? strndup((const char *)msg->event.exec.target->executable->path.data,
                                 msg->event.exec.target->executable->path.length)
                       : NULL;
        default:
            return NULL;
    }
}

static void handler(es_client_t *client, const es_message_t *msg) {
    (void)client;
    pid_t pid = audit_token_to_pid(msg->process->audit_token);
    if (g_stop || pid != g_target_pid) {
        return;
    }
    const char *path = message_path(msg);
    const char *name = event_name(msg->event_type);
    char row[MAX_ROW_BYTES];
    int written;
    if (path != NULL) {
        written = snprintf(row, sizeof(row),
                           "{\"event\":\"%s\",\"pid\":%d,\"path\":\"%s\",\"result\":\"notify\"}\n",
                           name, pid, path);
        free((void *)path);
    } else {
        written = snprintf(row, sizeof(row),
                           "{\"event\":\"%s\",\"pid\":%d,\"path\":null,\"result\":\"notify\"}\n",
                           name, pid);
    }
    if (written <= 0 || written >= (int)sizeof(row)) {
        g_overflow = 1;
        g_stop = 1;
        return;
    }
    if (g_row_count >= MAX_ROWS || g_row_bytes + (uint64_t)written > 16777216ULL) {
        g_overflow = 1;
        g_stop = 1;
        return;
    }
    fputs(row, g_out);
    g_row_count++;
    g_row_bytes += (uint64_t)written;
}

static void usage(void) {
    fprintf(stderr, "usage: es_read_isolation_audit --pid <worker-pid> --out <transcript.jsonl>\n");
}

int main(int argc, char **argv) {
    const char *out_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pid") == 0 && i + 1 < argc) {
            g_target_pid = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            usage();
            return 2;
        }
    }
    if (g_target_pid <= 0 || out_path == NULL) {
        usage();
        return 2;
    }
    g_out = fopen(out_path, "w");
    if (g_out == NULL) {
        fprintf(stderr, "cannot open transcript: %s\n", out_path);
        return 1;
    }
    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);

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
    };
    es_return_t sub = es_subscribe(client, subscribed,
                                   sizeof(subscribed) / sizeof(subscribed[0]));
    if (sub != ES_RETURN_SUCCESS) {
        fprintf(stderr, "es_subscribe failed: %d\n", sub);
        es_delete_client(client);
        fclose(g_out);
        return 1;
    }
    while (!g_stop) {
        pause();
    }
    es_delete_client(client);
    if (fflush(g_out) != 0 || fclose(g_out) != 0) {
        fprintf(stderr, "transcript finalize failed\n");
        return 1;
    }
    fprintf(stderr, "rows=%llu bytes=%llu overflow=%d\n",
            (unsigned long long)g_row_count, (unsigned long long)g_row_bytes,
            g_overflow);
    return g_overflow ? 3 : 0;
}
