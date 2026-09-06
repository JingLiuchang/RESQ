#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <sstream>
#include <vector>
#include <matrix.h>
#include <utils.h>
#include <ivf_rabitq.h>
#include <getopt.h>
#include "space.h"

using namespace std;

const int MAXK = 100;

long double rotation_time = 0;
int probe_base = 50;
char data_path[256] = "";
std::vector<int> nprobes_override;

template<uint64_t D, uint64_t B>
void test(const Matrix<float> &Q, const Matrix<float> &RandQ, const Matrix<unsigned> &G,
          const IVFRN<D, B> &ivf, int k) {
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;

    // ========================================================================
    // Search Parameter

    // ========================================================================
    std::vector<int> nprobes = nprobes_override;
    if (nprobes.empty()) {
        for (int i = 1; i <= 20; i++) nprobes.push_back(probe_base * i);
    }
    for (int nprobe : nprobes) {
        float total_time = 0;
        float total_ratio = 0;
        int correct = 0;
#ifdef COUNT_SCAN
        count_scan = 0;
        all_dist_count = 0;
#endif
        for (int i = 0; i < Q.n; i++) {
            GetCurTime(&run_start);
            ResultHeap KNNs = ivf.search(Q.data + i * Q.d, RandQ.data + i * RandQ.d, k, nprobe);
            GetCurTime(&run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;

            int tmp_correct = 0;
            while (KNNs.empty() == false) {
                int id = KNNs.top().second;
                KNNs.pop();
                for (int j = 0; j < k; j++)
                    if (id == G.data[i * G.d + j])tmp_correct++;
            }
            correct += tmp_correct;
        }
        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);

//        cout << "------------------------------------------------" << endl;
#ifdef COUNT_SCAN
        cout << "Count Full Scan " << count_scan << endl;
        cout << "All Distance Count " << all_dist_count << endl;
        cout << "Ratio:: " << (double) count_scan / all_dist_count << endl;
#endif
//        cout << "nprobe = " << nprobe << " k = " << k <<" Query Bits "<< B_QUERY << endl;
//        cout << "Recall = " << recall * 100.000 << "%\t" << "Ratio = " << average_ratio << endl;
//        cout << "Time = " << time_us_per_query << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
        cout << recall * 100.0 << " " << 1e6 / (time_us_per_query) << endl;
    }
}

#if defined(ESTIMATED_RERANK)
template<uint64_t D, uint64_t B>
void test_estimated_rerank(const Matrix<float> &Q, const Matrix<float> &RandQ, const Matrix<unsigned> &G,
                           const IVFRN<D, B> &ivf, int k, uint64_t rerank_size) {
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;

    // ========================================================================
    // Search Parameter

    // ========================================================================
    std::vector<int> nprobes = nprobes_override;
    if (nprobes.empty()) {
        for (int i = 1; i <= 20; i++) nprobes.push_back(probe_base * i);
    }
    for (int nprobe : nprobes) {
        float total_time = 0;
        float total_ratio = 0;
        int correct = 0;
#ifdef COUNT_SCAN
        count_scan = 0;
        all_dist_count = 0;
#endif
        for (int i = 0; i < Q.n; i++) {
            GetCurTime(&run_start);
            ResultHeap KNNs = ivf.search_estimated_rerank(
                    Q.data + i * Q.d,
                    RandQ.data + i * RandQ.d,
                    k,
                    nprobe,
                    rerank_size);
            GetCurTime(&run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;

            int tmp_correct = 0;
            while (KNNs.empty() == false) {
                int id = KNNs.top().second;
                KNNs.pop();
                for (int j = 0; j < k; j++)
                    if (id == G.data[i * G.d + j])tmp_correct++;
            }
            correct += tmp_correct;
        }
        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);

//        cout << "------------------------------------------------" << endl;
#ifdef COUNT_SCAN
        cout << "Count Full Scan " << count_scan << endl;
        cout << "All Distance Count " << all_dist_count << endl;
        cout << "Ratio:: " << (double) count_scan / all_dist_count << endl;
#endif
//        cout << "nprobe = " << nprobe << " k = " << k <<" Query Bits "<< B_QUERY << endl;
//        cout << "Recall = " << recall * 100.000 << "%\t" << "Ratio = " << average_ratio << endl;
//        cout << "Time = " << time_us_per_query << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
        cout << recall * 100.0 << " " << 1e6 / (time_us_per_query) << endl;
    }
}

template<uint64_t D, uint64_t B>
void test_dispatch(const Matrix<float> &Q, const Matrix<float> &RandQ, const Matrix<unsigned> &G,
                   const IVFRN<D, B> &ivf, int k, const std::string &search_mode, uint64_t rerank_size) {
    if (search_mode == "rerank") {
        test_estimated_rerank(Q, RandQ, G, ivf, k, rerank_size);
    } else {
        test(Q, RandQ, G, ivf, k);
    }
}
#define RUN_RABITQ_TEST(Q, RandQ, G, ivf, k) test_dispatch(Q, RandQ, G, ivf, k, search_mode, rerank_size)
#else
#define RUN_RABITQ_TEST(Q, RandQ, G, ivf, k) test(Q, RandQ, G, ivf, k)
#endif

int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",        no_argument,       0, 'h'},

            // Query Parameter
            {"K",           required_argument, 0, 'k'},

            // Indexing Path
            {"dataset",     required_argument, 0, 'd'},
            {"source",      required_argument, 0, 's'},
            {"result_path", required_argument, 0, 'r'},
            {"probes",      required_argument, 0, 'p'},
#if defined(ESTIMATED_RERANK)
            {"mode",        required_argument, 0, 'm'},
            {"rerank",      required_argument, 0, 'R'},
#endif
    };

    int ind, bit;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char result_path[256] = "";
    int subk = 0;
#if defined(ESTIMATED_RERANK)
    std::string search_mode = "inline";
    uint64_t rerank_size = 0;
    const char *optstring = "d:r:k:s:b:p:m:R:";
#else
    const char *optstring = "d:r:k:s:b:p:";
#endif

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, optstring, longopts, &ind);
        switch (iarg) {
            case 'k':
                if (optarg)subk = atoi(optarg);
                break;
            case 's':
                if (optarg)strcpy(source, optarg);
                break;
            case 'r':
                if (optarg)strcpy(result_path, optarg);
                break;
            case 'd':
                if (optarg)strcpy(dataset, optarg);
                break;
            case 'b':
                if (optarg) bit = atoi(optarg);
                break;
            case 'p':
                if (optarg) {
                    std::stringstream ss(optarg);
                    std::string tok;
                    while (std::getline(ss, tok, ',')) {
                        int v = std::stoi(tok);
                        if (v > 0) nprobes_override.push_back(v);
                    }
                }
                break;
#if defined(ESTIMATED_RERANK)
            case 'm':
                if (optarg) search_mode = optarg;
                break;
            case 'R':
                if (optarg) rerank_size = std::stoull(optarg);
                break;
#endif
        }
    }
#if defined(ESTIMATED_RERANK)
    if (search_mode != "inline" && search_mode != "rerank") {
        std::cerr << "Unsupported search mode: " << search_mode << std::endl;
        return 1;
    }
    if (search_mode == "rerank" && rerank_size == 0) {
        std::cerr << "Rerank mode requires -R/--rerank > 0" << std::endl;
        return 1;
    }
#endif
    // ================================================================================================================================
    // Data Files
    char query_path[256] = "";
    sprintf(query_path, "%s%s_query.fvecs", source, dataset);
    Matrix<float> Q(query_path);

    sprintf(data_path, "%s%s_base.fvecs", source, dataset);

    char groundtruth_path[256] = "";
    sprintf(groundtruth_path, "%s%s_groundtruth.ivecs", source, dataset);
    Matrix<unsigned> G(groundtruth_path);

    char transformation_path[256] = "";
    sprintf(transformation_path, "%sP_C%d_B%d.fvecs", source, numC, bit);
    Matrix<float> P(transformation_path);

    char index_path[256] = "";
    sprintf(index_path, "%sivfrabitq%d_B%d.index", source, numC, bit);
    std::cerr << index_path << std::endl;

#if defined(FAST_SCAN)
    char result_file_view[256] = "";
    sprintf(result_file_view, "%s%s_ivf_fast_scan.log", result_path, dataset, numC, bit);
#elif defined(SCAN)
    char result_file_view[256] = "";
    sprintf(result_file_view, "%s%s_ivfrabitq_scan.log", result_path, dataset, numC, bit);
#endif
    std::cerr << result_file_view << std::endl;
    std::cerr << "Loading Succeed!" << std::endl;
    // ================================================================================================================================


    freopen(result_file_view, "a", stdout);
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;
    GetCurTime(&run_start);

    Matrix<float> RandQ(Q.n, bit, Q);
    RandQ = mul(RandQ, P);

    GetCurTime(&run_end);
    GetTime(&run_start, &run_end, &usr_t, &sys_t);
    rotation_time = usr_t * 1e6 / Q.n;
    std::string str_data(dataset);
    std::cerr << "dataset:: " << str_data << std::endl;
#if defined(ESTIMATED_RERANK)
    std::cerr << "search_mode:: " << search_mode << " rerank_size:: " << rerank_size << std::endl;
#endif
    if (str_data == "msong") {
        const uint64_t BB = 448, DIM = 420;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 5;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "gist") {
        const uint64_t BB = 960, DIM = 960;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 25;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "deep1M") {
        const uint64_t BB = 256, DIM = 256;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "tiny5m") {
        const uint64_t BB = 384, DIM = 384;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 25;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "word2vec") {
        const uint64_t BB = 320, DIM = 300;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "sift") {
        const uint64_t BB = 128, DIM = 128;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 8;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "glove2.2m") {
        const uint64_t BB = 320, DIM = 300;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "OpenAI-1536") {
        const uint64_t BB = 1536, DIM = 1536;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "OpenAI-3072") {
        const uint64_t BB = 3072, DIM = 3072;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data.find("msmarc") != std::string::npos) {
        const uint64_t BB = 1024, DIM = 1024;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "yt1m") {
        const uint64_t BB = 1024, DIM = 1024;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    // ---- Benchmark datasets (added for PVLDB 2027 survey) ----------------
    if (str_data == "deep1M-96") {
        const uint64_t BB = 128, DIM = 96;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "laion") {
        const uint64_t BB = 512, DIM = 512;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 25;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "text2image") {
        const uint64_t BB = 256, DIM = 200;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "imagenet1m") {
        const uint64_t BB = 768, DIM = 768;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 20;
        RUN_RABITQ_TEST(Q, RandQ, G, ivf, subk);
    }
    // msmarco1M (d=1024) handled above by find("msmarc") with BB=1024, probe_base=30
    return 0;
}
