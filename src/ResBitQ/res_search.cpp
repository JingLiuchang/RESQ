
#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#define FAST_SCAN
#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <sstream>
#include <vector>
#include <matrix.h>
#include <utils.h>
#include "ivf_res.h"
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
          const IVFRES<D, B> &ivf, int k) {
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
//        cout << "nprobe = " << nprobe << " k = " << k << endl;
//        cout << "Recall = " << recall * 100.000 << "%\t" << "Ratio = " << average_ratio << endl;
//        cout << "Time = " << time_us_per_query << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
        cout << recall * 100.0 << " " << 1e6 / (time_us_per_query) << endl;
    }
}

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
    };

    int ind, bit;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char result_path[256] = "";
    int subk = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:r:k:s:b:c:p:", longopts, &ind);
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
            case 'c':
                if (optarg) numC = atoi(optarg);
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
        }
    }

    // ================================================================================================================================
    // Data Files
    char query_path[256] = "";
    sprintf(query_path, "%s%s_query.fvecs", source, dataset);
    Matrix<float> Q(query_path);

    char mean_path[256] = "";
    sprintf(mean_path, "%s%s_mean.fvecs", source, dataset);
    Matrix<float> M(mean_path);

    sprintf(data_path, "%s%s_proj.fvecs", source, dataset);

    char groundtruth_path[256] = "";
    sprintf(groundtruth_path, "%s%s_groundtruth.ivecs", source, dataset);
    Matrix<unsigned> G(groundtruth_path);

    char random_matrix_path[256] = "";
    sprintf(random_matrix_path, "%sRESP_C%d_B%d.fvecs", source, numC, bit);
    Matrix<float> P(random_matrix_path);

    char PCA_matrix_path[256] = "";
    sprintf(PCA_matrix_path, "%s%s_pca.fvecs", source, dataset);
    Matrix<float> PCA(PCA_matrix_path);

    char index_path[256] = "";
#ifdef RESIDUAL_SPLIT
    sprintf(index_path, "%sivf_split%ld_B%d.index", source, numC, bit);
#else
    sprintf(index_path, "%sivf_res%ld_B%d.index", source, numC, bit);
#endif

    std::cerr << index_path << std::endl;
    char result_file_view[256] = "";
#if defined(RESIDUAL_SPLIT)
    sprintf(result_file_view, "%s%s_ivf_split_scan_%ld_%d.log", result_path, dataset, numC, bit);
#else
    sprintf(result_file_view, "%s%s_ivf_res_scan_%ld_%d.log", result_path, dataset, numC, bit);
#endif
    std::cerr << result_file_view << std::endl;
    std::cerr << "Loading Succeed!" << std::endl;
    // ================================================================================================================================


    freopen(result_file_view, "a", stdout);
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;
    GetCurTime(&run_start);
    std::cerr << "begin Matrix Operation" << std::endl;
    Matrix<float> PCAQ(Q.n, Q.d, Q);
    PCAQ = mul(PCAQ, PCA);
    PCAQ = cen(PCAQ, M);
    auto TEMP_Q = resize_matrix(PCAQ, PCAQ.n, bit);
    auto RandQ = mul(TEMP_Q, P);

    GetCurTime(&run_end);
    GetTime(&run_start, &run_end, &usr_t, &sys_t);
    rotation_time = usr_t * 1e6 / Q.n;
    std::string str_data(dataset);
    // Strip "mrq_" prefix so existing dataset if-blocks match unchanged.
    // File paths still use the full dataset name (via the dataset[] char array).
    if (str_data.size() > 4 && str_data.substr(0, 4) == "mrq_")
        str_data = str_data.substr(4);
    std::cerr << "dataset:: " << str_data << std::endl;
    if (str_data == "msong") {
        const uint64_t BB = 128, DIM = 420;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 5;
        var_count = 5;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "gist") {
        const uint64_t BB = 320, DIM = 960;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 25;
        var_count = 5;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "deep1M") {
        const uint64_t BB = 128, DIM = 256;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        var_count = 5;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "tiny5m") {
        const uint64_t BB = 128, DIM = 384;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 25;
        var_count = 4;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "word2vec") {
        const uint64_t BB = 256, DIM = 300;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        var_count = 3;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "sift") {
        const uint64_t BB = 64, DIM = 128;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 50;
        var_count = 4;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "glove2.2m") {
        const uint64_t BB = 256, DIM = 300;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        var_count = 4;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "OpenAI-1536") {
        const uint64_t BB = 512, DIM = 1536;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        var_count = 20;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "OpenAI-3072") {
        const uint64_t BB = 512, DIM = 3072;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        var_count = 20;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data.find("msmarc") != std::string::npos) {
        const uint64_t BB = 320, DIM = 1024;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        var_count = 20;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "yt1m") {
        const uint64_t BB = 512, DIM = 1024;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        var_count = 10;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    // ---- Benchmark datasets (added for PVLDB 2027 survey) ----------------
    if (str_data == "deep1M-96") {
        const uint64_t BB = 64, DIM = 96;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        var_count = 4;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "laion") {
        const uint64_t BB = 256, DIM = 512;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 25;
        var_count = 5;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "text2image") {
        const uint64_t BB = 128, DIM = 200;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        var_count = 4;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    if (str_data == "imagenet1m") {
        const uint64_t BB = 384, DIM = 768;
        IVFRES<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 20;
        var_count = 5;
        test(PCAQ, RandQ, G, ivf, subk);
    }
    // msmarco1M (d=1024) handled above by find("msmarc") with BB=320, probe_base=30, var_count=20
    return 0;
}
