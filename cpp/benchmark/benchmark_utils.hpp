// cpp/benchmark/benchmark_utils.hpp
//
// Shared infrastructure for Phase 6: warm-up + repeated-run timing with
// mean/min/max/stddev (per SRS section 31 — "Warm-up runs: 5, Measured
// runs: 20... Mean, Minimum, Maximum, Standard deviation"), plus a minimal
// CSV writer for reports/benchmark_results.csv.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace gcv
{

    struct TimingStats
    {
        double mean_ms = 0.0;
        double min_ms = 0.0;
        double max_ms = 0.0;
        double stddev_ms = 0.0;
    };

    // Computes mean/min/max/stddev from a set of timing samples (in ms).
    // Factored out so callers that gather samples a different way (e.g. CUDA
    // event-based per-phase timing, rather than run_timed's host-clock wrapper)
    // can still get consistent statistics.
    inline TimingStats compute_stats(const std::vector<double> &samples)
    {
        TimingStats stats;
        stats.min_ms = *std::min_element(samples.begin(), samples.end());
        stats.max_ms = *std::max_element(samples.begin(), samples.end());

        double sum = 0.0;
        for (double s : samples)
            sum += s;
        stats.mean_ms = sum / samples.size();

        double sq_sum = 0.0;
        for (double s : samples)
            sq_sum += (s - stats.mean_ms) * (s - stats.mean_ms);
        stats.stddev_ms = std::sqrt(sq_sum / samples.size());

        return stats;
    }

    // Runs `fn` `warmup` times (results discarded — this absorbs CUDA context
    // initialization, cache warm-up, etc. so measured runs reflect steady-state
    // performance), then `measured` times with a host high-resolution clock
    // around each call. `fn` takes no arguments; callers capture whatever
    // state they need in a lambda.
    template <typename Func>
    TimingStats run_timed(Func &&fn, int warmup = 5, int measured = 20)
    {
        for (int i = 0; i < warmup; ++i)
        {
            fn();
        }

        std::vector<double> samples;
        samples.reserve(measured);
        for (int i = 0; i < measured; ++i)
        {
            const auto start = std::chrono::high_resolution_clock::now();
            fn();
            const auto end = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        return compute_stats(samples);
    }

    // Minimal append-only CSV writer. Caller writes the header once, then one
    // row per (algorithm, resolution, ...) combination.
    class CsvWriter
    {
    public:
        explicit CsvWriter(const std::string &path) : out_(path) {}

        bool is_open() const { return out_.is_open(); }

        void write_header(const std::vector<std::string> &columns)
        {
            write_row(columns);
        }

        void write_row(const std::vector<std::string> &values)
        {
            for (size_t i = 0; i < values.size(); ++i)
            {
                out_ << values[i];
                if (i + 1 < values.size())
                    out_ << ",";
            }
            out_ << "\n";
        }

    private:
        std::ofstream out_;
    };

} // namespace gcv