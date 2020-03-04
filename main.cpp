#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "lib/eigen-eigen-323c052e1731/Eigen/Core"
#include "lib/eigen-eigen-323c052e1731/Eigen/LU"

using Real = double;

const size_t data_size = 506;
const size_t dim = 13;
const Real rho= 1.0;
const Real lambda = 1.0;
const Real n = 10;
const size_t max_iter = 1000;

Eigen::Matrix<Real, 10, 1> soft_threshold(Eigen::Matrix<Real, 10, 1> xs) {
    const Real threshold = lambda / rho;

    Eigen::Matrix<Real, 10, 1> rs;
    for (int i=0; i<10; i++) {
        Real x = xs[i];
        if (x > threshold) {
            rs[i] = x - threshold;
        } else if (-threshold <= x && x <= threshold) {
            rs[i] = 0;
        } else {
            rs[i] = -x + threshold;
        }
    }
    return xs;
}

void soft_threshold(Real threshold, Eigen::VectorXd& x) {
    for(int i=0; i<dim; i++) {
        if (x(i) > threshold) {
            x(i) = x(i) - threshold;
        } else if (-threshold <= x(i) && x(i) <= threshold) {
            x(i) = 0;
        } else if (x(i) < -threshold) {
            x(i) = x(i) + threshold;
        } else {
            throw;
        }
    }
}

std::vector<std::string> split(const char delimiter, const std::string s) {
    std::vector<std::string> ret;
    std::string buf;
    for (size_t i=0; i<s.size(); i++) {
        if (s[i] == ' ') {
            // SKIP
        } if (s[i] == delimiter) {
            ret.push_back(buf);
            buf.clear();
        } else {
            buf.push_back(s[i]);
        }
    }
    if (!buf.empty()) ret.push_back(buf);
    return ret;
}

int main() {
    Eigen::Matrix<Real, data_size, dim> A; // 説明変数
    Eigen::VectorXd b(data_size); // 目的変数
    Eigen::Matrix<Real, dim, dim> I; I.setIdentity();
    Eigen::VectorXd x(dim); // ハイパーパラメータ
    Eigen::VectorXd y(dim);
    Eigen::VectorXd z(dim);
    Eigen::VectorXd zeros(dim); zeros.setZero();

    // 説明変数の読み込み
    std::ifstream data("/home/eliza/Project/admm_lasso/dataset.csv");
    for(int i=0; i<data_size; i++) {
        std::string line; getline(data, line);
        auto cnt = 0;
        for (std::string s: split(',', line))
            A(i, cnt++) = stod(s);
    }

    // 目的変数の読み込み
    std::ifstream target("/home/eliza/Project/admm_lasso/ans.csv");
    for(int i=0; i<data_size; i++) {
        std::string line;
        getline(target, line);
        b(i) = stod(line);
    }

    // データの正規化
    Eigen::Matrix<double, 1, dim> mean = A.colwise().mean();
    Eigen::Matrix<double, data_size, dim> centered = A.rowwise() - mean;
    Eigen::Matrix<double, data_size, dim> sq = centered.array().square();
    Eigen::Matrix<double, 1, dim> variance = sq.colwise().sum() / data_size;
    Eigen::Matrix<double, data_size, dim> std;
    for (int i=0; i<dim; i++) {
        for (int l=0; l < data_size; l++) {
            std(l, i) = A(l, i) / variance(i);
        }
    }

    A = std;

    // 初期化
    x = (A.transpose() * b).adjoint() / data_size;
    z = x;
    y = zeros;

    // 学習
    for (int iter_cnt=1; iter_cnt<=max_iter; iter_cnt++) {
        Eigen::Matrix<Real, dim, dim> inv_matrix = ((A.transpose() * A).adjoint()/data_size + rho*I.adjoint()).inverse();
        Eigen::Matrix<Real, dim, 1> matrix = (A.transpose()*b).adjoint()/data_size + rho*z.adjoint() - y.adjoint();
        x = inv_matrix * matrix;

        z = x.adjoint() + y.adjoint()/rho;
        soft_threshold(lambda/rho, z);

        y = y.adjoint() + (rho * (x - z).adjoint());
    }

    // しきい値より小さいハイパーパラメータを0に
    for (int i=0; i<dim; i++) {
        x(i) = abs(x(i)) < 1e-7 ? 0 : x(i);
    }
    std::cout << "params: " << x.adjoint() << std::endl;
}
