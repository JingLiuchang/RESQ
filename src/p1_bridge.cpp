#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

thread_local std::string last_error;

class MrqP1Context {
  public:
    MrqP1Context(
        std::size_t input_dimension,
        std::size_t projection_dimension,
        const float *projected_center,
        const float *rotation)
        : input_dimension_(input_dimension), projection_dimension_(projection_dimension) {
        if (input_dimension == 0 || projection_dimension == 0 || projection_dimension > input_dimension) {
            throw std::invalid_argument("MRQ dimensions must satisfy 0 < projection <= input");
        }
        if (projected_center == nullptr || rotation == nullptr) {
            throw std::invalid_argument("projected_center and rotation must be non-null");
        }
        projected_center_.assign(projected_center, projected_center + projection_dimension_);
        rotation_.assign(rotation, rotation + projection_dimension_ * projection_dimension_);
        code_bytes_ = (projection_dimension_ + 7) / 8;
        inverse_sqrt_projection_ = 1.0f / std::sqrt(static_cast<float>(projection_dimension_));
    }

    double encode(
        const float *vectors,
        const float *projected_vectors,
        std::size_t vector_count,
        std::size_t input_dimension,
        std::size_t projection_dimension,
        std::size_t thread_count) {
        validate_dimensions(input_dimension, projection_dimension);
        if ((vectors == nullptr || projected_vectors == nullptr) && vector_count != 0) {
            throw std::invalid_argument("encode inputs must be non-null");
        }
        set_threads(thread_count);
        encoded_count_ = vector_count;
        binary_codes_.assign(vector_count * code_bytes_, 0);
        projected_norms_.assign(vector_count, 0.0f);
        residual_squared_norms_.assign(vector_count, 0.0f);
        x0_.assign(vector_count, 0.0f);

        const auto start = std::chrono::steady_clock::now();
#pragma omp parallel
        {
            std::vector<float> centered(projection_dimension_);
            std::vector<float> normalized(projection_dimension_);
            std::vector<float> rotated(projection_dimension_);
#pragma omp for schedule(static)
            for (std::int64_t row_signed = 0;
                 row_signed < static_cast<std::int64_t>(vector_count);
                 ++row_signed) {
                const std::size_t row = static_cast<std::size_t>(row_signed);
                const float *vector = vectors + row * input_dimension_;
                const float *projected = projected_vectors + row * projection_dimension_;
                double vector_squared_norm = 0.0;
                double projected_squared_norm = 0.0;
                double centered_squared_norm = 0.0;
                for (std::size_t dim = 0; dim < input_dimension_; ++dim) {
                    vector_squared_norm += static_cast<double>(vector[dim]) * vector[dim];
                }
                for (std::size_t dim = 0; dim < projection_dimension_; ++dim) {
                    projected_squared_norm += static_cast<double>(projected[dim]) * projected[dim];
                    centered[dim] = projected[dim] - projected_center_[dim];
                    centered_squared_norm += static_cast<double>(centered[dim]) * centered[dim];
                }

                const float projected_norm = static_cast<float>(std::sqrt(centered_squared_norm));
                projected_norms_[row] = projected_norm;
                residual_squared_norms_[row] = static_cast<float>(vector_squared_norm - projected_squared_norm);
                const float inverse_norm = projected_norm > 0.0f ? 1.0f / projected_norm : 0.0f;
                for (std::size_t dim = 0; dim < projection_dimension_; ++dim) {
                    normalized[dim] = centered[dim] * inverse_norm;
                }
                rotate(normalized.data(), rotated.data());

                float inner = 0.0f;
                std::uint8_t *code = binary_codes_.data() + row * code_bytes_;
                for (std::size_t dim = 0; dim < projection_dimension_; ++dim) {
                    const bool positive = rotated[dim] > 0.0f;
                    if (positive) {
                        code[dim >> 3U] |= static_cast<std::uint8_t>(1U << (dim & 7U));
                    }
                    inner += (positive ? 1.0f : -1.0f) * rotated[dim] * inverse_sqrt_projection_;
                }
                if (!std::isfinite(inner)) inner = 0.8f;
                if (inner == 0.0f) inner = 1e-6f;
                x0_[row] = inner;
            }
        }
        const auto stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(stop - start).count();
    }

    void distances(
        const float *queries,
        const float *projected_queries,
        std::size_t query_count,
        std::size_t input_dimension,
        std::size_t projection_dimension,
        const std::int64_t *candidate_indices,
        std::size_t candidates_per_query,
        std::size_t thread_count,
        float *output) const {
        validate_dimensions(input_dimension, projection_dimension);
        const std::size_t pair_count = query_count * candidates_per_query;
        if ((queries == nullptr || projected_queries == nullptr || candidate_indices == nullptr || output == nullptr) &&
            pair_count != 0) {
            throw std::invalid_argument("distance inputs must be non-null");
        }
        if (encoded_count_ == 0 && pair_count != 0) {
            throw std::runtime_error("encode must be called before distances");
        }
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            const std::int64_t index = candidate_indices[pair];
            if (index < 0 || static_cast<std::size_t>(index) >= encoded_count_) {
                throw std::out_of_range("candidate index is outside the encoded base set");
            }
        }
        set_threads(thread_count);

#pragma omp parallel
        {
            std::vector<float> centered(projection_dimension_);
            std::vector<float> normalized(projection_dimension_);
            std::vector<float> rotated(projection_dimension_);
#pragma omp for schedule(static)
            for (std::int64_t query_signed = 0;
                 query_signed < static_cast<std::int64_t>(query_count);
                 ++query_signed) {
                const std::size_t query_index = static_cast<std::size_t>(query_signed);
                const float *query = queries + query_index * input_dimension_;
                const float *projected = projected_queries + query_index * projection_dimension_;
                double query_squared_norm = 0.0;
                double projected_squared_norm = 0.0;
                double centered_squared_norm = 0.0;
                for (std::size_t dim = 0; dim < input_dimension_; ++dim) {
                    query_squared_norm += static_cast<double>(query[dim]) * query[dim];
                }
                for (std::size_t dim = 0; dim < projection_dimension_; ++dim) {
                    projected_squared_norm += static_cast<double>(projected[dim]) * projected[dim];
                    centered[dim] = projected[dim] - projected_center_[dim];
                    centered_squared_norm += static_cast<double>(centered[dim]) * centered[dim];
                }
                const float query_projected_norm = static_cast<float>(std::sqrt(centered_squared_norm));
                const float inverse_norm = query_projected_norm > 0.0f ? 1.0f / query_projected_norm : 0.0f;
                for (std::size_t dim = 0; dim < projection_dimension_; ++dim) {
                    normalized[dim] = centered[dim] * inverse_norm;
                }
                rotate(normalized.data(), rotated.data());
                const double query_residual_squared_norm = query_squared_norm - projected_squared_norm;

                for (std::size_t candidate = 0; candidate < candidates_per_query; ++candidate) {
                    const std::size_t pair = query_index * candidates_per_query + candidate;
                    const std::size_t data_index = static_cast<std::size_t>(candidate_indices[pair]);
                    const std::uint8_t *code = binary_codes_.data() + data_index * code_bytes_;
                    double inner = 0.0;
                    for (std::size_t dim = 0; dim < projection_dimension_; ++dim) {
                        const bool positive = ((code[dim >> 3U] >> (dim & 7U)) & 1U) != 0;
                        inner += (positive ? 1.0 : -1.0) * rotated[dim] * inverse_sqrt_projection_;
                    }
                    const double data_projected_norm = projected_norms_[data_index];
                    output[pair] = static_cast<float>(
                        data_projected_norm * data_projected_norm +
                        query_projected_norm * query_projected_norm +
                        residual_squared_norms_[data_index] +
                        query_residual_squared_norm -
                        2.0 * data_projected_norm * query_projected_norm * inner / x0_[data_index]);
                }
            }
        }
    }

    std::size_t code_bits() const {
        return projection_dimension_;
    }

    std::size_t memory_bytes() const {
        return code_bytes_ + 3 * sizeof(float);
    }

  private:
    void validate_dimensions(std::size_t input_dimension, std::size_t projection_dimension) const {
        if (input_dimension != input_dimension_ || projection_dimension != projection_dimension_) {
            throw std::invalid_argument("dimensions do not match the MRQ context");
        }
    }

    void rotate(const float *input, float *output) const {
        for (std::size_t row = 0; row < projection_dimension_; ++row) {
            double value = 0.0;
            const float *rotation_row = rotation_.data() + row * projection_dimension_;
            for (std::size_t column = 0; column < projection_dimension_; ++column) {
                value += static_cast<double>(input[column]) * rotation_row[column];
            }
            output[row] = static_cast<float>(value);
        }
    }

    static void set_threads(std::size_t thread_count) {
#ifdef _OPENMP
        omp_set_num_threads(static_cast<int>(std::max<std::size_t>(1, thread_count)));
#else
        (void)thread_count;
#endif
    }

    std::size_t input_dimension_;
    std::size_t projection_dimension_;
    std::size_t code_bytes_ = 0;
    std::size_t encoded_count_ = 0;
    float inverse_sqrt_projection_ = 0.0f;
    std::vector<float> projected_center_;
    std::vector<float> rotation_;
    std::vector<std::uint8_t> binary_codes_;
    std::vector<float> projected_norms_;
    std::vector<float> residual_squared_norms_;
    std::vector<float> x0_;
};

template <typename Function>
int guarded_call(Function &&function) noexcept {
    try {
        last_error.clear();
        function();
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    } catch (...) {
        last_error = "unknown C++ exception";
        return -1;
    }
}

} // namespace

extern "C" {

const char *mrq_p1_last_error() noexcept {
    return last_error.c_str();
}

void *mrq_p1_create(
    std::size_t input_dimension,
    std::size_t projection_dimension,
    const float *projected_center,
    const float *rotation) noexcept {
    try {
        last_error.clear();
        return new MrqP1Context(
            input_dimension, projection_dimension, projected_center, rotation);
    } catch (const std::exception &error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "unknown C++ exception";
        return nullptr;
    }
}

void mrq_p1_destroy(void *context) noexcept {
    delete static_cast<MrqP1Context *>(context);
}

int mrq_p1_encode(
    void *context,
    const float *vectors,
    const float *projected_vectors,
    std::size_t vector_count,
    std::size_t input_dimension,
    std::size_t projection_dimension,
    std::size_t thread_count,
    double *elapsed_seconds) noexcept {
    return guarded_call([&] {
        if (context == nullptr || elapsed_seconds == nullptr) {
            throw std::invalid_argument("context and elapsed_seconds must be non-null");
        }
        *elapsed_seconds = static_cast<MrqP1Context *>(context)->encode(
            vectors,
            projected_vectors,
            vector_count,
            input_dimension,
            projection_dimension,
            thread_count);
    });
}

int mrq_p1_distances(
    void *context,
    const float *queries,
    const float *projected_queries,
    std::size_t query_count,
    std::size_t input_dimension,
    std::size_t projection_dimension,
    const std::int64_t *candidate_indices,
    std::size_t candidates_per_query,
    std::size_t thread_count,
    float *output) noexcept {
    return guarded_call([&] {
        if (context == nullptr) {
            throw std::invalid_argument("context must be non-null");
        }
        static_cast<MrqP1Context *>(context)->distances(
            queries,
            projected_queries,
            query_count,
            input_dimension,
            projection_dimension,
            candidate_indices,
            candidates_per_query,
            thread_count,
            output);
    });
}

std::size_t mrq_p1_code_bits(void *context) noexcept {
    return context == nullptr ? 0 : static_cast<MrqP1Context *>(context)->code_bits();
}

std::size_t mrq_p1_memory_bytes(void *context) noexcept {
    return context == nullptr ? 0 : static_cast<MrqP1Context *>(context)->memory_bytes();
}

} // extern "C"
