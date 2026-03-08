#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace NNUE {
    constexpr int INPUT_SIZE = 768;
    constexpr int HIDDEN_SIZE = 32;
    constexpr int QA = 255;
    constexpr int QB = 64;
    constexpr int NNUE_SCALE = 400;

    struct Network {
        alignas(64) i16 feature_weights[INPUT_SIZE][HIDDEN_SIZE];
        alignas(64) i16 feature_bias[HIDDEN_SIZE];
        i16 output_weights[2 * HIDDEN_SIZE];
        i16 output_bias;
    };

    static Network g_net {};
    static bool g_loaded = false;
    static std::string g_loaded_path;

    static inline i16 read_i16_le(const uint8_t* p) {
        uint16_t u = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
        return static_cast<i16>(u);
    }

    static inline i32 screlu(i32 x) {
        x = std::clamp(x, 0, QA);
        return x * x;
    }

    static bool load_from_buffer(const uint8_t* data, size_t size) {
        if (!data || !size) {
            g_loaded = false;
            return false;
        }

        constexpr size_t L0W_BYTES = static_cast<size_t>(INPUT_SIZE) * HIDDEN_SIZE * sizeof(i16);
        constexpr size_t L0B_BYTES = static_cast<size_t>(HIDDEN_SIZE) * sizeof(i16);
        constexpr size_t L1W_BYTES = static_cast<size_t>(2 * HIDDEN_SIZE) * sizeof(i16);
        constexpr size_t L1B_BYTES = sizeof(i16);
        constexpr size_t MIN_BYTES = L0W_BYTES + L0B_BYTES + L1W_BYTES + L1B_BYTES;

        if (size < MIN_BYTES) {
            g_loaded = false;
            return false;
        }

        size_t off = 0;

        for (int feat = 0; feat < INPUT_SIZE; ++feat)
            for (int h = 0; h < HIDDEN_SIZE; ++h) {
                g_net.feature_weights[feat][h] = read_i16_le(&data[off]);
                off += sizeof(i16);
            }

        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            g_net.feature_bias[h] = read_i16_le(&data[off]);
            off += sizeof(i16);
        }

        for (int i = 0; i < 2 * HIDDEN_SIZE; ++i) {
            g_net.output_weights[i] = read_i16_le(&data[off]);
            off += sizeof(i16);
        }

        g_net.output_bias = read_i16_le(&data[off]);
        off += sizeof(i16);

        g_loaded = true;
        return true;
    }

    static bool load_from_file(const std::string& path) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin)
            return false;

        fin.seekg(0, std::ios::end);
        std::streamsize size = fin.tellg();
        fin.seekg(0, std::ios::beg);

        if (size <= 0)
            return false;

        std::vector<uint8_t> buf(static_cast<size_t>(size));
        if (!fin.read(reinterpret_cast<char*>(buf.data()), size))
            return false;

        if (!load_from_buffer(buf.data(), buf.size()))
            return false;

        g_loaded_path = path;
        return true;
    }

    bool init(i32 verbose) {
        if (g_loaded)
            return true;

        std::vector<std::string> paths;

        if (const char* env = std::getenv("C4KE_NNUE_FILE"))
            paths.emplace_back(env);

        paths.emplace_back("nnue.bin");

        for (const auto& path : paths)
            if (load_from_file(path)) {
                if (verbose)
                    std::cout << "info string NNUE loaded: " << g_loaded_path << std::endl;
                return true;
            }

        if (verbose)
            std::cout << "info string NNUE not found, using classical eval" << std::endl;

        return false;
    }

    bool is_ready() {
        return g_loaded;
    }

    static inline i32 pov_square(const Board& board, i32 sq) {
        return board.stm == WHITE ? sq : sq ^ 56;
    }

    static inline void chess768_indices(i32 pt, i32 sq, i32 is_opponent, i32& stm_idx, i32& ntm_idx) {
        const i32 pc = 64 * pt;
        const i32 sq_flip = sq ^ 56;

        stm_idx = (is_opponent ? 384 : 0) + pc + sq;
        ntm_idx = (is_opponent ? 0 : 384) + pc + sq_flip;
    }

    i32 evaluate(const Board& board) {
        if (!g_loaded)
            return 0;

        i32 acc_stm[HIDDEN_SIZE];
        i32 acc_ntm[HIDDEN_SIZE];

        for (int i = 0; i < HIDDEN_SIZE; ++i)
            acc_stm[i] = acc_ntm[i] = static_cast<i32>(g_net.feature_bias[i]);

        for (i32 type = PAWN; type <= KING; ++type) {
            u64 bb = board.pieces[type];

            while (bb) {
                const i32 sq_abs = LSB(bb);
                bb &= bb - 1;

                const i32 piece_color = !!(board.colors[BLACK] & (1ull << sq_abs));
                const i32 sq = pov_square(board, sq_abs);
                const i32 is_opponent = piece_color != board.stm;

                i32 stm_idx = 0, ntm_idx = 0;
                chess768_indices(type, sq, is_opponent, stm_idx, ntm_idx);

                const i16* w_stm = g_net.feature_weights[stm_idx];
                const i16* w_ntm = g_net.feature_weights[ntm_idx];

                for (int h = 0; h < HIDDEN_SIZE; ++h) {
                    acc_stm[h] += static_cast<i32>(w_stm[h]);
                    acc_ntm[h] += static_cast<i32>(w_ntm[h]);
                }
            }
        }

        long long out = 0;

        for (int h = 0; h < HIDDEN_SIZE; ++h)
            out += static_cast<long long>(screlu(acc_stm[h])) * static_cast<long long>(g_net.output_weights[h]);

        for (int h = 0; h < HIDDEN_SIZE; ++h)
            out += static_cast<long long>(screlu(acc_ntm[h])) * static_cast<long long>(g_net.output_weights[HIDDEN_SIZE + h]);

        out /= QA;
        out += static_cast<long long>(g_net.output_bias);
        out *= NNUE_SCALE;
        out /= static_cast<long long>(QA) * static_cast<long long>(QB);

        return static_cast<i32>(out);
    }
}
