#include <algorithm>
#include <iterator>
#include <functional>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include "cnpy.h"
#include <mpi.h>
//---------------------------------------------------------------
constexpr auto deg = 3;
constexpr auto N   = 300;
constexpr auto rho = 1.204;

constexpr auto dt     = 5e-3;
constexpr auto margin = 0.3;

constexpr auto Ndof  = deg*N;
constexpr auto Ninv  = 1.0/N;
constexpr auto SKIN2 = (margin*0.5) * (margin*0.5);
constexpr auto dt2   = dt*0.5;
constexpr auto dt4   = dt*0.25;
constexpr auto N_A   = N*4/5;

const double Lbox = std::pow(N/rho, 1.0/deg);
const double Linv = 1.0/Lbox;

double conf[N][deg], velo[N][deg], force[N][deg], NL_config[N][deg];
int point[N], list[N*100];
double vxi1 = 0.0;

enum {X, Y, Z};
//---------------------------------------------------------------
void init_lattice() {
    const auto ln   = std::ceil(std::pow(N, 1.0/deg));
    const auto haba = Lbox/ln;
    const auto lnz  = std::ceil(N/(ln*ln));
    const auto zaba = Lbox/lnz;

    for (int i=0; i<N; i++) {
        int iz = std::floor(i/(ln*ln));
        int iy = std::floor((i - iz*ln*ln)/ln);
        int ix = i - iz*ln*ln - iy*ln;

        conf[i][X] = haba*0.5 + haba * ix;
        conf[i][Y] = haba*0.5 + haba * iy;
        conf[i][Z] = zaba*0.5 + zaba * iz;

        for (int d=0; d<deg; d++) {
            conf[i][d] -= Lbox * std::round(conf[i][d] * Linv);
        }
    }
}
inline void remove_drift() {
    double vel1 = 0.0, vel2 = 0.0, vel3 = 0.0;
    for (int i=0; i<N; i++) {
        vel1 += velo[i][X];
        vel2 += velo[i][Y];
        vel3 += velo[i][Z];
    }
    vel1 *= Ninv;
    vel2 *= Ninv;
    vel3 *= Ninv;
    for (int i=0; i<N; i++) {
        velo[i][X] -= vel1;
        velo[i][Y] -= vel2;
        velo[i][Z] -= vel3;
    }
}
void init_vel_MB(const double T_targ, std::mt19937 &mt) {
    std::normal_distribution<double> dist_trans(0.0, std::sqrt(T_targ));
    for (int i=0; i<N; i++) {
        velo[i][X] = dist_trans(mt);
        velo[i][Y] = dist_trans(mt);
        velo[i][Z] = dist_trans(mt);
    }
    remove_drift();
}
void init_species(std::mt19937 &mt) {
    std::vector<int> v(N);
    std::iota(v.begin(), v.end(), 0);
    std::shuffle(v.begin(), v.end(), mt);
    for (int i=0; i<N; i+=2) {
        const int id0 = v[i];
        const int id1 = v[i+1];
        const double xid0 = conf[id0][X];
        const double yid0 = conf[id0][Y];
        const double zid0 = conf[id0][Z];
        const double xid1 = conf[id1][X];
        const double yid1 = conf[id1][Y];
        const double zid1 = conf[id1][Z];
        conf[id0][X] = xid1;
        conf[id0][Y] = yid1;
        conf[id0][Z] = zid1;
        conf[id1][X] = xid0;
        conf[id1][Y] = yid0;
        conf[id1][Z] = zid0;
    }
}
//---------------------------------------------------------------
inline double KABLJ_energy(const int &ki, const int &kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 1.0;
        case 1:
            return 1.5;
        }
    case 1:
        switch (kj) {
        case 0:
            return 1.5;
        case 1:
            return 0.5;
        }
    }
    return 0.0;
}
inline double KABLJ_sij(const int &ki, const int &kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 1.0;
        case 1:
            return 0.8;
        }
    case 1:
        switch (kj) {
        case 0:
            return 0.8;
        case 1:
            return 0.88;
        }
    }
    return 0.0;
}
//---------------------------------------------------------------
void generate_NL() {
    int nlist = -1;
    for (int i=0; i<N; i++) {
        point[i] = nlist+1;
        const int ki = i>=N_A;
        for (int j=i+1; j<N; j++) {
            const int kj = j>=N_A;
            double dx = conf[i][X] - conf[j][X];
            double dy = conf[i][Y] - conf[j][Y];
            double dz = conf[i][Z] - conf[j][Z];
            dx -= Lbox * floor(dx * Linv + 0.5);
            dy -= Lbox * floor(dy * Linv + 0.5);
            dz -= Lbox * floor(dz * Linv + 0.5);

            const double rij2   = dx*dx + dy*dy + dz*dz;
            const double sij1   = KABLJ_sij(ki, kj);
            const double rlist1 = 2.5*sij1 + margin;
            const double rlist2 = rlist1 * rlist1;
            if (rij2 < rlist2) {
                nlist++;
                list[nlist] = j;
            }
        }
    }
    std::copy(*conf, *conf+Ndof, *NL_config);
}
void calc_force() {
    std::fill(*force, *force+Ndof, 0.0);
    for (int i=0; i<N-1; i++) {
        const int pend = point[i+1];
        if (pend == point[i]) continue;
        const int ki = i>=N_A;
        for (int p=point[i]; p<pend; p++) {
            const int j = list[p];
            const int kj = j>=N_A;

            double dx = conf[i][X] - conf[j][X];
            double dy = conf[i][Y] - conf[j][Y];
            double dz = conf[i][Z] - conf[j][Z];
            dx -= Lbox * floor(dx * Linv + 0.5);
            dy -= Lbox * floor(dy * Linv + 0.5);
            dz -= Lbox * floor(dz * Linv + 0.5);

            double rij2  = dx*dx + dy*dy + dz*dz;
            double rcut1 = 2.5 * KABLJ_sij(ki, kj);
            if (rij2 < rcut1 * rcut1) {
                double rij1 = sqrt(rij2);
                double rij6 = rij2 * rij2 * rij2;
                double rij12 = rij6 * rij6;

                double sij1 = KABLJ_sij(ki, kj);
                double sij2 = sij1 * sij1;
                double sij6 = sij2 * sij2 * sij2;
                double sij12 = sij6 * sij6;

                double dV = (2.0*sij12 - sij6*rij6)/(rij12 * rij2) - (2.0 - 2.5*2.5*2.5*2.5*2.5*2.5)/(2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5 * sij1 * rij1);
                dV *= -24.0 * KABLJ_energy(ki, kj);

                force[i][X] -= dV * dx;
                force[i][Y] -= dV * dy;
                force[i][Z] -= dV * dz;
                force[j][X] += dV * dx;
                force[j][Y] += dV * dy;
                force[j][Z] += dV * dz;
            }
        }
    }
}
//---------------------------------------------------------------
double calc_potential() {
    double ans = 0.0;
    for (int i=0; i<N-1; i++) {
        const int ki = i>=N_A;
        const int pend = point[i+1];
        if (pend == point[i]) continue;
        for (int p=point[i]; p<pend; p++) {
            const int j = list[p];
            const int kj = j>=N_A;

            double dx = conf[i][X] - conf[j][X];
            double dy = conf[i][Y] - conf[j][Y];
            double dz = conf[i][Z] - conf[j][Z];
            dx -= Lbox * floor(dx * Linv + 0.5);
            dy -= Lbox * floor(dy * Linv + 0.5);
            dz -= Lbox * floor(dz * Linv + 0.5);

            double rij2 = dx*dx + dy*dy + dz*dz;
            double rcut1 = 2.5 * KABLJ_sij(ki, kj);
            if (rij2 < rcut1 * rcut1) {
                double sij1 = KABLJ_sij(ki, kj);
                double sij2 = sij1 * sij1;
                double sij6 = sij2 * sij2 * sij2;
                double sij12 = sij6 * sij6;

                double rij1 = sqrt(rij2);
                double rij6 = rij2 * rij2 * rij2;
                double rij12 = rij6 * rij6;

                double V = (sij12 - rij6*sij6)/rij12 - (1.0 - 2.5*2.5*2.5*2.5*2.5*2.5)/(2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5) + 6.0*(2.0 - 2.5*2.5*2.5*2.5*2.5*2.5)/(2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5) * (rij1/sij1 - 2.5);
                ans += 4.0*KABLJ_energy(ki, kj) * V;
            }
        }
    }
    return ans;
}
double calc_potential_N2() {
    double ans = 0.0;
    for (int i=0; i<N; i++) {
        const int ki = i>=N_A;
        for (int j=i+1; j<N; j++) {
            const int kj = j>=N_A;

            double dx = conf[i][X] - conf[j][X];
            double dy = conf[i][Y] - conf[j][Y];
            double dz = conf[i][Z] - conf[j][Z];
            dx -= Lbox * floor(dx * Linv + 0.5);
            dy -= Lbox * floor(dy * Linv + 0.5);
            dz -= Lbox * floor(dz * Linv + 0.5);

            double rij2 = dx*dx + dy*dy + dz*dz;
            double rcut1 = 2.5 * KABLJ_sij(ki, kj);
            if (rij2 < rcut1 * rcut1) {
                double sij1 = KABLJ_sij(ki, kj);
                double sij2 = sij1 * sij1;
                double sij6 = sij2 * sij2 * sij2;
                double sij12 = sij6 * sij6;

                double rij1 = sqrt(rij2);
                double rij6 = rij2 * rij2 * rij2;
                double rij12 = rij6 * rij6;

                double V = (sij12 - rij6*sij6)/rij12 - (1.0 - 2.5*2.5*2.5*2.5*2.5*2.5)/(2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5) + 6.0*(2.0 - 2.5*2.5*2.5*2.5*2.5*2.5)/(2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5*2.5) * (rij1/sij1 - 2.5);
                ans += 4.0*KABLJ_energy(ki, kj) * V;
            }
        }
    }
    return ans;
}
//---------------------------------------------------------------
inline void velocity_update() {
    for (int i=0; i<N; i++) {
        velo[i][X] += dt2*force[i][X];
        velo[i][Y] += dt2*force[i][Y];
        velo[i][Z] += dt2*force[i][Z];
    }
}
inline void position_update() {
    for (int i=0; i<N; i++) {
        conf[i][X] += dt*velo[i][X];
        conf[i][Y] += dt*velo[i][Y];
        conf[i][Z] += dt*velo[i][Z];
    }
}
inline void PBC() {
    for (int i=0; i<N; i++) {
        conf[i][X] -= Lbox * floor(conf[i][X] * Linv + 0.5);
        conf[i][Y] -= Lbox * floor(conf[i][Y] * Linv + 0.5);
        conf[i][Z] -= Lbox * floor(conf[i][Z] * Linv + 0.5);
    }  
}
inline void NL_check() {
    double dev_max = 0.0;
    for (int i=0; i<N; i++) {
        double xij = conf[i][X] - NL_config[i][X];
        double yij = conf[i][Y] - NL_config[i][Y];
        double zij = conf[i][Z] - NL_config[i][Z];
        xij -= Lbox * floor(xij * Linv + 0.5);
        yij -= Lbox * floor(yij * Linv + 0.5);
        zij -= Lbox * floor(zij * Linv + 0.5);

        dev_max = std::max(dev_max, (xij*xij + yij*yij + zij*zij));
    }
    if (dev_max > SKIN2) {// renew neighbor list
        generate_NL();
    }
}
//---------------------------------------------------------------
void print_log(long t) {
    double K = 0.5*std::inner_product(*velo, *velo+Ndof, *velo, 0.0);
    double U = calc_potential();
    std::cout << std::setprecision(6) << std::scientific
              << dt*t << ","
              << K*Ninv << ","
              << U*Ninv << "," << (K+U)*Ninv << std::endl;
}
void NVE(const double tsim) {
    calc_force();
    // for logging ///////////////////////////////////
    const auto logbin = std::pow(10.0, 1.0/9);
    int counter = 5;
    auto checker = 1e-3 * std::pow(logbin, counter);
    //////////////////////////////////////////////////

    long t = 0;
    print_log(t);
    const long steps = tsim/dt;
    while (t < steps) {
        velocity_update();
        position_update();
        PBC();
        NL_check();
        calc_force();
        velocity_update();

        t++;
        if (dt*t > checker) {
            checker *= logbin;
            print_log(t);
        }
    }
}
//---------------------------------------------------------------
void NVT(const double T_targ, const double tsim) {
    calc_force();
    // Nose-Hoover variables
    const auto gkBT = Ndof*T_targ;

    long t = 0;
    const long steps = tsim/dt;
    while (t < steps) {
        // Nose-Hoover chain (QMASS = 1.0)
        double uk = std::inner_product(*velo, *velo+Ndof, *velo, 0.0);
        vxi1 += dt4 * (uk - gkBT);
        double temp = std::exp(-vxi1 * dt2);
        std::transform(*velo, *velo+Ndof+N, *velo,
                       std::bind(std::multiplies<double>(), std::placeholders::_1, temp));
        vxi1 += dt4 * (uk*temp*temp - gkBT);

        velocity_update();
        position_update();
        PBC();
        NL_check();
        calc_force();
        velocity_update();

        // Nose-Hoover chain (QMASS = 1.0)
        uk    = std::inner_product(*velo, *velo+Ndof, *velo, 0.0);
        vxi1 += dt4 * (uk - gkBT);
        temp  = std::exp(-vxi1 * dt2);
        std::transform(*velo, *velo+Ndof+N, *velo,
                       std::bind(std::multiplies<double>(), std::placeholders::_1, temp));
        vxi1 += dt4 * (uk*temp*temp - gkBT);

        t++;
        if (!(t & 127)) remove_drift();
    }
}
//---------------------------------------------------------------
int main(int argc, char** argv) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::stringstream ss;
    ss << std::setfill('0') << std::right << std::setw(3) << rank;
    std::string name = ss.str()+".npz";
    cnpy::NpyArray arr2 = cnpy::npz_load(name, "position");
    std::vector<double> conf2(arr2.data<double>(), arr2.data<double>() + Ndof);
    std::copy(conf2.begin(), conf2.end(), *conf);

    cnpy::NpyArray arr3 = cnpy::npz_load(name, "velocity");
    std::vector<double> conf3(arr3.data<double>(), arr3.data<double>() + Ndof);
    std::copy(conf3.begin(), conf3.end(), *velo);
    generate_NL();

    NVE(1e5);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
