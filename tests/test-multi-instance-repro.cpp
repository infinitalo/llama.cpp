// Multi-instance memory-behaviour repro.
//
// model: Llama-3.2-1B-Instruct-Q4_0.gguf
// Usage:
//   ./test-multi-instance-repro -m /path/to/model.gguf -ngl 999 -c 1024 -n 1
//
// Add --buffer-load to run Scenario 3: QVAC-style buffer load cycles.
// Each cycle reads the file into a vector<uint8_t>, loads the model (moving
// the buffer in), keeps a separate live copy of the raw bytes until after
// inference (simulating Node.js GC not immediately collecting the JS wrapper),
// then frees model+ctx first, then drops the raw copy.

#include "arg.h"
#include "common.h"
#include "llama-cpp.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static long read_kb(const char * key) {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long val = -1;
    const size_t key_len = strlen(key);
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, key_len) == 0) {
            sscanf(line + key_len, " %ld kB", &val);
            break;
        }
    }
    fclose(f);
    return val;
}

static void print_mem(const char * label) {
    long rss  = read_kb("VmRSS:");
    long size = read_kb("VmSize:");
    long peak = read_kb("VmPeak:");
    long swap = read_kb("VmSwap:");
    if (rss < 0) {
        printf("[MEM] %-36s (no /proc/self/status)\n", label);
        return;
    }
    printf("[MEM] %-36s RSS=%5ld MiB  VmSize=%5ld MiB  VmPeak=%5ld MiB  VmSwap=%5ld MiB\n",
           label, rss / 1024, size / 1024, peak / 1024, swap / 1024);
    fflush(stdout);
}

static bool decode_once(llama_context * ctx, const llama_model * model) {
    const auto * vocab = llama_model_get_vocab(model);
    llama_token tok = llama_vocab_bos(vocab);
    if (tok == LLAMA_TOKEN_NULL) {
        // model has no BOS; pick any valid token
        tok = 0;
    }
    llama_batch batch = llama_batch_get_one(&tok, 1);
    return llama_decode(ctx, batch) == 0;
}

static std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "failed to open %s\n", path.c_str());
        return {};
    }
    const size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char *>(buf.data()), sz);
    return buf;
}

int main(int argc, char ** argv) {
    // --buffer-load: opt-in to Scenario 3 (QVAC-style buffer load cycles).
    // Strip it before handing argv to common_params_parse.
    bool do_buffer_load = false;
    {
        int out = 1;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--buffer-load") == 0) {
                do_buffer_load = true;
            } else {
                argv[out++] = argv[i];
            }
        }
        argc = out;
    }

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    print_mem("baseline");

    printf("\n=== Scenario 1: two simultaneous instances ===\n");
    {
        common_params p1 = params;
        common_params p2 = params;

        print_mem("S1 before load #1");
        common_init_result_ptr i1 = common_init_from_params(p1);
        if (!i1 || !i1->model() || !i1->context()) {
            LOG_ERR("S1: failed to load instance #1\n");
            return 1;
        }
        print_mem("S1 after  load #1");

        common_init_result_ptr i2 = common_init_from_params(p2);
        if (!i2 || !i2->model() || !i2->context()) {
            LOG_ERR("S1: failed to load instance #2\n");
            return 1;
        }
        print_mem("S1 after  load #2");

        if (!decode_once(i1->context(), i1->model())) {
            LOG_ERR("S1: decode #1 failed\n");
            return 1;
        }
        print_mem("S1 after  decode #1");
        if (!decode_once(i2->context(), i2->model())) {
            LOG_ERR("S1: decode #2 failed\n");
            return 1;
        }
        print_mem("S1 after  decode #2");

        i1.reset();
        print_mem("S1 after  unload #1");
        i2.reset();
        print_mem("S1 after  unload #2");
    }
    print_mem("S1 after  scope exit");

    printf("\n=== Scenario 2: 20 load/unload cycles ===\n");
    constexpr int NUM_CYCLES = 20;
    for (int i = 1; i <= NUM_CYCLES; ++i) {
        common_params p = params;
        char label[64];

        snprintf(label, sizeof(label), "S2 cycle %d before load", i);
        print_mem(label);

        common_init_result_ptr inst = common_init_from_params(p);
        if (!inst || !inst->model() || !inst->context()) {
            LOG_ERR("S2: cycle %d failed to load\n", i);
            return 1;
        }
        snprintf(label, sizeof(label), "S2 cycle %d after  load", i);
        print_mem(label);

        if (!decode_once(inst->context(), inst->model())) {
            LOG_ERR("S2: cycle %d decode failed\n", i);
            return 1;
        }
        snprintf(label, sizeof(label), "S2 cycle %d after  decode", i);
        print_mem(label);

        inst.reset();
        snprintf(label, sizeof(label), "S2 cycle %d after  unload", i);
        print_mem(label);
    }

    if (do_buffer_load) {
        printf("\n=== Scenario 3: QVAC-style buffer load/unload cycles ===\n");
        printf("    (file read -> model load -> context create -> decode -> free model+ctx -> drop raw bytes)\n\n");
        constexpr int BL_CYCLES = 20;
        for (int i = 1; i <= BL_CYCLES; ++i) {
            char label[80];

            // Step 1: read the whole file into memory (FilesystemDL equivalent)
            std::vector<uint8_t> raw = read_file(params.model.path);
            if (raw.empty()) {
                fprintf(stderr, "S3: cycle %d failed to read file\n", i);
                return 1;
            }
            snprintf(label, sizeof(label), "S3 cycle %d after  file read", i);
            print_mem(label);
            // Keep a second copy alive to simulate the JS wrapper holding the
            // original allocation while llama.cpp moves it into the model.
            std::vector<uint8_t> raw_copy = raw;

            // Step 2: load model (moves raw; raw_copy lives on alongside model)
            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = params.n_gpu_layers >= 0 ? params.n_gpu_layers : 0;
            mparams.use_mmap     = false;
            llama_model * model = llama_model_load_from_buffer(std::move(raw), mparams);
            if (!model) {
                fprintf(stderr, "S3: cycle %d failed to load model\n", i);
                return 1;
            }
            snprintf(label, sizeof(label), "S3 cycle %d after  model load", i);
            print_mem(label);

            // Step 3: create context
            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx   = params.n_ctx > 0 ? (uint32_t) params.n_ctx : 1024;
            cparams.n_batch = 512;
            llama_context * ctx = llama_init_from_model(model, cparams);
            if (!ctx) {
                fprintf(stderr, "S3: cycle %d failed to create context\n", i);
                llama_model_free(model);
                return 1;
            }
            snprintf(label, sizeof(label), "S3 cycle %d after  ctx create", i);
            print_mem(label);

            // Step 4: one decode (exercises GPU kernel launch + buffer use)
            if (!decode_once(ctx, model)) {
                fprintf(stderr, "S3: cycle %d decode failed (non-fatal)\n", i);
            }
            snprintf(label, sizeof(label), "S3 cycle %d after  decode", i);
            print_mem(label);

            // Step 5: free model+ctx synchronously (C++ side)
            llama_free(ctx);
            llama_model_free(model);
            snprintf(label, sizeof(label), "S3 cycle %d after  free model+ctx", i);
            print_mem(label);

            // Step 6: drop the raw copy (simulates GC eventually collecting
            // the JS wrapper that was keeping the file buffer alive)
            raw_copy.clear();
            raw_copy.shrink_to_fit();
            snprintf(label, sizeof(label), "S3 cycle %d after  drop raw copy", i);
            print_mem(label);
        }
    }

    llama_backend_free();
    print_mem("after backend free");
    return 0;
}
