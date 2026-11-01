// HTTP handling of the typed decision endpoints: POST /v1/systemone and POST /v1/decisions, one
// read per question (per label digit for a wide choice) on the served chat model.
#include "ninfer/request.h"
#include "serve/http_server.h"
#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/request_events.h"
#include "serve/request_validation.h"
#include "serve/systemone.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::serve {

void HttpServer::handle_decisions(const httplib::Request& req, httplib::Response& res,
                                  DecisionDialect dialect) {
    const bool system_one = dialect == DecisionDialect::SystemOne;
    const char* protocol  = system_one ? "systemone" : "openai_decisions";
    DecisionRequest request;
    ModelRegistry::Lease lease;
    std::vector<DecisionLabels> labels;
    // Each question's reads (decision_read_nodes): its label characters as the model's tokens,
    // each one token.
    std::vector<std::vector<DecisionReadNode>> nodes;
    std::vector<GenerationRequest> questions;
    std::optional<GenerationRequest> conversation;
    // Reads of media run as requests that resume from the first read's state in the context
    // cache: a branch's prompt is its tokens alone.
    bool has_media = false;
    std::map<std::string, ninfer::TokenId> resolved;
    const auto token = [&](const std::string& piece) {
        auto found = resolved.find(piece);
        if (found == resolved.end()) {
            const std::vector<ninfer::TokenId> ids = lease.service().tokenize(piece, false);
            if (ids.size() != 1) {
                bad_request("label " + piece + " is not one token of the model's tokenizer",
                            "questions");
            }
            found = resolved.emplace(piece, ids.front()).first;
        }
        return found->second;
    };
    try {
        const RequestJson body = parse_json_body(req);
        request       = system_one ? parse_systemone_request(body) : parse_decisions_request(body);
        lease         = lease_model(request.model, req);
        request.model = lease.model();
        if (request.conversation) {
            try {
                conversation =
                    parse_chat_completion_request(*request.conversation, request_limits(options_))
                        .generation;
            } catch (ApiException& exception) {
                ApiError error = exception.error();
                error.param    = "conversation." + error.param;
                throw ApiException(std::move(error));
            }
        }
        has_media =
            !request.media.empty() || (conversation && conversation->media_item_count() != 0);
        if (has_media) {
            if (!lease.service().options().enable_vision) {
                ApiError error;
                error.status  = 501;
                error.type    = "not_implemented_error";
                error.param   = request.media.empty() ? "conversation.messages" : "images";
                error.message = "the served model reads no images or video";
                throw ApiException(std::move(error));
            }
            // Every read's request holds its own copy of the media: Runoff's reads included.
            std::size_t bytes = 0;
            for (const ContentPart& part : request.media) { bytes += part.source.value.size(); }
            if (conversation) {
                for (const ChatTurn& turn : conversation->messages) {
                    for (const ContentPart& part : turn.content) {
                        if (part.kind != ContentKind::Text) { bytes += part.source.value.size(); }
                    }
                }
            }
            std::size_t copies = request.questions.size();
            if (request.runoff) {
                for (const DecisionQuestion& question : request.questions) {
                    if (question.type == DecisionType::Choice) {
                        copies += 2 * question.options.size();
                    }
                }
            }
            if (bytes * copies > kMaximumDecisionMediaCopies) {
                bad_request("the media's data times the reads exceeds " +
                                std::to_string(kMaximumDecisionMediaCopies >> 20) +
                                " MiB: ask fewer questions about them per request",
                            "questions");
            }
        }
        for (const DecisionQuestion& question : request.questions) {
            labels.push_back(decision_labels(question));
            nodes.push_back(decision_read_nodes(labels.back(), token));
            if (labels.back().width > 1) {
                if (has_media) {
                    // Its later digits are read from token prompts, which carry no media.
                    bad_request("a choice of more than " + std::to_string(kMaximumLetterOptions) +
                                    " options cannot be asked about images or video",
                                "questions");
                }
                // A wide label must read as its digits' tokens one after another.
                const std::string& sample = labels.back().texts.back();
                std::vector<ninfer::TokenId> digits;
                for (const char digit : sample) { digits.push_back(token(std::string(1, digit))); }
                if (lease.service().tokenize(sample, false) != digits) {
                    bad_request("the model's tokenizer does not read " + sample +
                                    " a digit at a time",
                                "questions");
                }
            }
            questions.push_back(make_decision_read(request, question, nodes.back().front().readout,
                                                   conversation ? &*conversation : nullptr));
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    GenerationService& service = lease.service();
    const auto cancelled       = [&req] { return client_disconnected(req); };
    // A read is never continued, so its own continuation is the first the cache evicts, and it
    // starts no session lineage.
    const ninfer::ContextCacheHints read_cache{
        .retention            = ninfer::CacheRetentionHint::Disposable,
        .update_session_index = false,
    };
    const RequestLogMetadata metadata{.model = request.model};
    int input_tokens  = 0;
    int output_tokens = 0;
    // Runs `reads` and returns their first tokens' records in order. With the Engine's branched
    // reads the state runs once, up to its cache boundary, and each read runs as a block over it in
    // shared passes. Otherwise (no branched reads, or media) the first read runs alone and leaves
    // the state in the context cache, and the rest resume from it a lane's worth at a time, each
    // window submitted as one group.
    const auto run_reads = [&](std::span<const BranchRead> reads) {
        std::vector<ninfer::TokenLogprob> records;
        records.reserve(reads.size());
        std::optional<PreparedRequest> branched;
        if (service.runs_branches() && !has_media) {
            try {
                branched.emplace(
                    service.submit_branches(reads, {.phase_timings = true}, cancelled, read_cache));
            } catch (const std::length_error&) {
                // A read longer past the state than one shared pass holds: the windows read them.
            }
        }
        if (branched) {
            PreparedRequest& prepared = *branched;
            const std::shared_ptr<RequestLifecycle> lifecycle =
                begin_request(make_request_log_context(++request_seq_, protocol,
                                                       *reads.front().request, metadata, prepared),
                              prepared);
            GenerationOutcome outcome;
            try {
                outcome = service.run(prepared, nullptr, cancelled);
            } catch (const ApiException& exception) {
                lifecycle->failure(make_generation_request_failure(exception.error()));
                throw;
            }
            lifecycle->done(outcome);
            if (outcome.branches.size() != reads.size()) {
                throw std::logic_error(
                    "the branched read returned a record count unlike its reads");
            }
            for (ninfer::BranchResult& branch : outcome.branches) {
                input_tokens += static_cast<int>(branch.prompt_tokens);
                output_tokens += 1;
                records.push_back(std::move(branch.first));
            }
            return records;
        }
        const std::size_t lanes =
            std::max<std::size_t>(1, static_cast<std::size_t>(service.options().max_concurrency));
        for (std::size_t begin = 0; begin < reads.size();) {
            const std::size_t end =
                std::min(begin == 0 ? std::size_t{1} : begin + lanes, reads.size());
            std::vector<StagedRequest> staged;
            for (std::size_t index = begin; index < end; ++index) {
                staged.push_back(service.stage_read(reads[index], {.phase_timings = true},
                                                    cancelled, read_cache));
            }
            std::vector<PreparedRequest> batch = service.submit_group(std::move(staged));
            std::vector<std::shared_ptr<RequestLifecycle>> lifecycles;
            for (std::size_t offset = 0; offset < batch.size(); ++offset) {
                lifecycles.push_back(begin_request(make_request_log_context(
                    ++request_seq_, protocol, *reads[begin + offset].request, metadata,
                    batch[offset]), batch[offset]));
            }
            for (std::size_t offset = 0; offset < batch.size(); ++offset) {
                GenerationOutcome outcome;
                try {
                    outcome = service.run(batch[offset], nullptr, cancelled);
                } catch (const ApiException& exception) {
                    lifecycles[offset]->failure(make_generation_request_failure(exception.error()));
                    throw;
                }
                lifecycles[offset]->done(outcome);
                input_tokens += outcome.prompt_tokens;
                output_tokens += outcome.completion_tokens;
                if (outcome.content_logprobs.empty()) {
                    throw std::runtime_error("the decision read produced no answer token");
                }
                records.push_back(std::move(outcome.content_logprobs.front()));
            }
            begin = end;
        }
        return records;
    };
    // The reads of the questions `which` names, in order, and the first read of each.
    const auto reads_of = [&](std::span<const std::size_t> which, std::vector<std::size_t>& first) {
        std::vector<BranchRead> reads;
        for (const std::size_t q : which) {
            first.push_back(reads.size());
            for (const DecisionReadNode& node : nodes[q]) {
                reads.push_back(BranchRead{.request      = &questions[q],
                                           .continuation = node.continuation,
                                           .readout      = node.readout});
            }
        }
        return reads;
    };
    const auto answer_of = [&](std::size_t q, std::span<const ninfer::TokenLogprob> records,
                               std::size_t first) {
        std::vector<const ninfer::TokenLogprob*> pointers;
        for (std::size_t n = 0; n < nodes[q].size(); ++n) {
            pointers.push_back(&records[first + n]);
        }
        return read_decision(pointers, nodes[q], labels[q]);
    };
    const GenerationRequest* conversation_read = conversation ? &*conversation : nullptr;
    std::vector<DecisionAnswer> answers(request.questions.size());
    try {
        // Questions read directly, and choice questions Runoff answers.
        std::vector<std::size_t> direct;
        std::vector<std::size_t> runoff;
        for (std::size_t q = 0; q < request.questions.size(); ++q) {
            const bool by_runoff =
                request.runoff && request.questions[q].type == DecisionType::Choice;
            (by_runoff ? runoff : direct).push_back(q);
        }
        // One run: the direct questions' reads and Runoff's pointwise reads (Consider).
        std::vector<std::size_t> first;
        std::vector<BranchRead> reads = reads_of(direct, first);
        DecisionQuestion noul;
        noul.type    = DecisionType::Noul;
        noul.options = {{.value = RequestJson("yes"), .text = "yes", .description = std::nullopt},
                        {.value = RequestJson("no"), .text = "no", .description = std::nullopt}};
        const DecisionLabels noul_labels               = decision_labels(noul);
        const std::vector<DecisionReadNode> noul_nodes = decision_read_nodes(noul_labels, token);
        std::size_t judged_reads                       = 0;
        for (const std::size_t q : runoff) { judged_reads += request.questions[q].options.size(); }
        std::vector<GenerationRequest> judgements;
        judgements.reserve(judged_reads);
        std::vector<std::size_t> judged_first;
        for (const std::size_t q : runoff) {
            judged_first.push_back(reads.size());
            const DecisionQuestion& question = request.questions[q];
            for (const std::size_t option : runoff_order(question)) {
                judgements.push_back(
                    make_runoff_pointwise(request, question, question.options[option],
                                          noul_nodes.front().readout, conversation_read));
                reads.push_back(BranchRead{.request = &judgements.back(),
                                           .readout = noul_nodes.front().readout});
            }
        }
        const std::vector<ninfer::TokenLogprob> records = run_reads(reads);
        for (std::size_t k = 0; k < direct.size(); ++k) {
            answers[direct[k]] = answer_of(direct[k], records, first[k]);
        }
        if (!runoff.empty()) {
            // Choose: each question's candidates in listings where each takes every position.
            DecisionRequest listed_request = request;
            listed_request.layout          = DecisionLayout::StateFirst;
            std::vector<RunoffCandidates> candidates;
            std::vector<DecisionQuestion> listings;
            std::vector<DecisionLabels> listing_labels;
            std::vector<std::vector<DecisionReadNode>> listing_nodes;
            std::size_t listing_count = 0;
            for (std::size_t k = 0; k < runoff.size(); ++k) {
                const DecisionQuestion& question = request.questions[runoff[k]];
                std::vector<DecisionAnswer> judged;
                for (std::size_t n = 0; n < question.options.size(); ++n) {
                    judged.push_back(read_decision(records[judged_first[k] + n], noul_labels,
                                                   noul_nodes.front().readout));
                }
                candidates.push_back(runoff_consider(question, judged));
                listing_count += candidates.back().ranking.size();
            }
            listings.reserve(listing_count);
            std::vector<GenerationRequest> listing_reads;
            listing_reads.reserve(listing_count);
            std::vector<BranchRead> second;
            std::vector<std::size_t> listing_first;
            for (std::size_t k = 0; k < runoff.size(); ++k) {
                const DecisionQuestion& question = request.questions[runoff[k]];
                for (std::size_t shift = 0; shift < candidates[k].ranking.size(); ++shift) {
                    listings.push_back(runoff_listing(question, candidates[k], shift));
                    listing_labels.push_back(decision_labels(listings.back()));
                    listing_nodes.push_back(decision_read_nodes(listing_labels.back(), token));
                    listing_reads.push_back(make_decision_read(listed_request, listings.back(),
                                                               listing_nodes.back().front().readout,
                                                               conversation_read));
                    listing_first.push_back(second.size());
                    for (const DecisionReadNode& node : listing_nodes.back()) {
                        second.push_back(BranchRead{.request      = &listing_reads.back(),
                                                    .continuation = node.continuation,
                                                    .readout      = node.readout});
                    }
                }
            }
            const std::vector<ninfer::TokenLogprob> listed = run_reads(second);
            std::size_t listing                            = 0;
            for (std::size_t k = 0; k < runoff.size(); ++k) {
                std::vector<DecisionAnswer> listing_answers;
                for (std::size_t shift = 0; shift < candidates[k].ranking.size(); ++shift) {
                    std::vector<const ninfer::TokenLogprob*> pointers;
                    for (std::size_t n = 0; n < listing_nodes[listing].size(); ++n) {
                        pointers.push_back(&listed[listing_first[listing] + n]);
                    }
                    listing_answers.push_back(
                        read_decision(pointers, listing_nodes[listing], listing_labels[listing]));
                    ++listing;
                }
                answers[runoff[k]] =
                    runoff_choose(request.questions[runoff[k]], candidates[k], listing_answers);
            }
        }
        // A near tie is read again alone, so its answer does not depend on what shared its pass;
        // the branch memo holds the record read beside the others, so the read passes it by.
        for (const std::size_t q : direct) {
            if (answers[q].margin >= request.recheck_margin) { continue; }
            const std::array<std::size_t, 1> one{q};
            std::vector<std::size_t> alone_first;
            std::vector<BranchRead> alone = reads_of(one, alone_first);
            for (BranchRead& read : alone) { read.memo = false; }
            const std::vector<ninfer::TokenLogprob> again = run_reads(alone);
            answers[q]                                    = answer_of(q, again, 0);
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_decision_response(request, answers, input_tokens, output_tokens),
                    "application/json");
}

} // namespace ninfer::serve
