#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// accept rules: "exact", "escape", "lam:<x>", "esc+<x>" (x in (0, 1)), "all". Throws std::invalid_argument
common_spec_accept common_spec_accept_from_str(const std::string & s);
std::string        common_spec_accept_to_str(const common_spec_accept & a);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// True if deferred catch-up rows plus one first-draft anchor per sequence fit in one llama_decode.
// Used by draft-mtp so a full-prefill stash (n_tokens == n_batch) does not add a 33rd row.
bool common_speculative_mtp_first_decode_fits(int32_t n_batch, int32_t catchup_rows, int32_t n_anchors);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

// one decoded MTP draft step. The step that stopped drafting is included; its token was not added
struct common_speculative_trace_step {
    llama_token  id;       // the drafted token (the top-1 on the stop step)
    float        q;        // the draft's probability of id: the sampled q, else the top-1 p
    float        p1;       // top-1 p of the step's candidates (T 1)
    float        H;        // entropy (nats) of the step's candidates
    const char * stop;     // "p_min", "iso", "n_max" or nullptr
};

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // > 0: the MTP draft samples at the target's temperature and top-p instead of taking the argmax
    // dists then holds the distribution of each drafted token, for common_sampler_sample_and_accept_n_dist
    float sample_temp  = 0.0f;
    float sample_top_p = 1.0f;

    // the request accepts drafts by an approximate rule (common_spec_accept::approx): the MTP draft then samples each
    // token with the request's own settings (these, plus sample_temp and sample_top_p) and drafts without p-min
    bool         accept_approx     = false;
    int32_t      target_top_k      = 0;
    float        target_min_p      = 0.0f;
    float        target_present    = 0.0f;
    int32_t      target_pen_last_n = 0;
    llama_tokens pen_window        = {}; // the target's penalty window up to and including id_last, oldest first

    std::vector<std::vector<llama_token_data>> dists = {};

    // when set, the MTP draft clears it and appends one entry per decoded step (--spec-trace)
    std::vector<common_speculative_trace_step> * trace = nullptr;
};

// the stop rule tables: piecewise-linear iso_q / iso_p1 over the points x, and theta
struct common_spec_stop_iso {
    float              theta = 0.0f;
    std::vector<float> x, q, p1;

    float lookup(const std::vector<float> & tab, float v) const;
};

// the built-in tables (P100 fork, Qwen3.8-27B, thinking-off; placeholders), or path's, theta overriding when > 0. Throws on a bad file
common_spec_stop_iso common_spec_stop_iso_load(const std::string & path, float theta);

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// true if any implementation requires the target's multi-layer tap capture
// (see llama_set_capture_layers / llama_get_embeddings_capture_ith) -- used by
// dspark, which conditions on several intermediate target layers concatenated
// per position rather than a single pre/post-norm embedding.
bool common_speculative_need_embd_capture(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// types of the implementations that were actually initialized, in priority order
std::vector<enum common_speculative_type> common_speculative_get_types(const common_speculative * spec);

// TEST/DEBUG ONLY: directly stage target-tap context rows for the dspark
// implementation (if registered), bypassing the normal process()-driven
// capture path, which requires a real target context with
// llama_set_capture_layers engaged and logits requested on every row. Used by
// the Phase 2 synthetic-target harness (tests/test-dspark-loop.cpp) to drive
// the block-draft loop deterministically without a target model.
// `feat` is [n_rows * n_embd_cap] row-major, `pos` is [n_rows] absolute
// positions, both appended to the sequence's pending context buffer exactly
// as process() would have. Returns false if no dspark implementation is
// registered.
bool common_speculative_dspark_stage_ctx_test(common_speculative * spec,
                                              llama_seq_id         seq_id,
                                              const float *        feat,
                                              int64_t              n_rows,
                                              int64_t              n_embd_cap,
                                              const int32_t *      pos);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
