#include <omp.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <chrono>

using namespace std;

const int NX = 5;
const int NY = 5;
const int NZ = 5;

void jacobi(int seed) {
    vector<vector<vector<double>>> u(NX, vector<vector<double>>(NY, vector<double>(NZ, 0.0)));
    vector<vector<vector<double>>> u_new(NX, vector<vector<double>>(NY, vector<double>(NZ, 0.0)));

    srand(seed);
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                u[i][j][k] = static_cast<double>(rand()) / RAND_MAX;
            }
        }
    }

    const int num_iterations = 1000;
    double start_time = omp_get_wtime();

    #pragma omp parallel for collapse(3)
    for (int iter = 0; iter < num_iterations; ++iter) {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                for (int k = 0; k < NZ; ++k) {
                    u_new[i][j][k] = (u[(i+1)%NX][j][k] + u[(i-1)%NX][j][k] +
                                      u[i][(j+1)%NY][k] + u[i][(j-1)%NY][k] +
                                      u[i][j][(k+1)%NZ] + u[i][j][(k-1)%NZ]) / 6.0;
                }
            }
        }

        swap(u, u_new);
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    long long total_ops = NX * NY * NZ * num_iterations * 6;
    double mflops = static_cast<double>(total_ops) / (time_taken * 1e6);

    cout << "MFLOPS: " << mflops << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <seed>" << endl;
        return 1;
    }

    int seed = atoi(argv[1]);
    jacobi(seed);

    return 0;
