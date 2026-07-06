#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <bitset>
#include <vector>

#include <algorithm> // for std::min
#include <iostream>

template <typename T> class Matrix2d {
  private:
    size_t rows, cols;
    std::vector<T> data;

  public:
    Matrix2d(size_t r, size_t c, T init_val = T())
        : rows(r), cols(c), data(r * c, init_val) {}

    Matrix2d(py::array_t<T, py::array::c_style> arr) {
        rows = arr.shape(0);
        cols = arr.shape(1);

        data.resize(rows * cols);

        const T *src = arr.data();
        std::copy(src, src + (rows * cols), data.begin());
    }

    // No folds, no lambdas. Pure, simple math the compiler loves.
    T &operator()(size_t r, size_t c) { return data[r * cols + c]; }

    const T &operator()(size_t r, size_t c) const { return data[r * cols + c]; }

    size_t dim(size_t index) { return index == 0 ? rows : cols; }

    const size_t dim(size_t index) const { return index == 0 ? rows : cols; }

    T *data_ptr() { return data.data(); }
};

Matrix2d<uint8_t> elementwise_AND_2d(const Matrix2d<uint8_t> &image1,
                                     const Matrix2d<uint8_t> &image2) {
    Matrix2d<uint8_t> result = image1;

    size_t num_rows = image1.dim(0);
    size_t num_cols = image1.dim(1);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            result(i, j) = (result(i, j) && image2(i, j));
        }
    }

    return result;
}

Matrix2d<uint8_t> elementwise_AND_2d(const Matrix2d<uint8_t> &image1,
                                     const Matrix2d<uint8_t> &image2,
                                     const Matrix2d<uint8_t> &image3) {
    Matrix2d<uint8_t> result = image1;

    size_t num_rows = image1.dim(0);
    size_t num_cols = image1.dim(1);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            result(i, j) = result(i, j) && image2(i, j) && image3(i, j);
        }
    }

    return result;
}

template <typename T> class Matrix3d {
    size_t dim1, dim2, dim3;
    std::vector<T> data;

  public:
    Matrix3d(size_t d1, size_t d2, size_t d3, T init_val = T())
        : dim1(d1), dim2(d2), dim3(d3), data(d1 * d2 * d3, init_val) {}

    // No folds, no lambdas. Pure, simple math the compiler loves.
    T &operator()(size_t i, size_t j, size_t k) {
        return data[i * dim2 * dim3 + j * dim3 + k];
    }

    const T &operator()(size_t i, size_t j, size_t k) const {
        return data[i * dim2 * dim3 + j * dim3 + k];
    }

    size_t dim(size_t index) {
        if (index == 0)
            return dim1;
        else if (index == 1)
            return dim2;
        else
            return dim3;
    }

    const size_t dim(size_t index) const {
        if (index == 0)
            return dim1;
        else if (index == 1)
            return dim2;
        else
            return dim3;
    }

    T *data_ptr() { return data.data(); }
};

int ec_binary_image_2d_naive(Matrix2d<uint8_t> &image) {
    size_t num_rows = image.dim(0);
    size_t num_cols = image.dim(1);

    int v = 0;
    int e = 0;
    int f = 0;

    for (size_t j = 1; j < num_cols - 1; ++j) {
        // iterate through pixels at the top edge
        f += image(0, j);
        v += image(0, j) || image(0, j + 1); // top-right vertex
        e += image(0, j);                    // upper edge
        e += image(0, j) || image(0, j + 1); // right edge
    }

    for (size_t i = 1; i < num_rows - 1; ++i) {
        // iterate through pixels at right edge
        f += image(i, num_cols - 1);
        v += image(i, num_cols - 1) ||
             image(i - 1, num_cols - 1); // top-right vertex
        e += image(i, num_cols - 1) || image(i - 1, num_cols - 1); // upper edge
        e += image(i, num_cols - 1);                               // right edge
    }

    for (size_t i = 1; i < num_rows - 1; ++i) {
        // iterate through pixels at left edge
        f += image(i, 0);

        v += image(i, 0) || image(i - 1, 0); // top-left vertex
        e += image(i, 0);                    // left edge

        v += image(i, 0) || image(i - 1, 0) || image(i - 1, 1) ||
             image(i, 1);                    // upper right vertex
        e += image(i, 0) || image(i - 1, 0); // upper edge
        e += image(i, 0) || image(i, 1);     // right edge
    }

    for (size_t j = 1; j < num_cols - 1; ++j) {
        // iterate through pixels at the bottom edge
        f += image(num_rows - 1, j);

        v += image(num_rows - 1, j) ||
             image(num_rows - 1, j + 1); // bottom-right vertex
        e += image(num_rows - 1, j);     // bottom edge

        v += image(num_rows - 1, j) || image(num_rows - 2, j) ||
             image(num_rows - 2, j + 1) ||
             image(num_rows - 1, j + 1); // upper right vertex
        e += image(num_rows - 1, j) || image(num_rows - 2, j);     // upper edge
        e += image(num_rows - 1, j) || image(num_rows - 1, j + 1); // right edge
    }

    // top-left pixel
    f += image(0, 0);
    v += image(0, 0) || image(0, 1); // top-right vertex
    e += 2 * image(0, 0);            // upper edge + left edge
    e += image(0, 0) || image(0, 1); // right edge
    v += image(0, 0);                // top-left vertex

    // top-right pixel
    f += image(0, num_cols - 1);
    v += image(0, num_cols - 1);     // top-right vertex
    e += 2 * image(0, num_cols - 1); // top edge + right edge

    // bottom-left pixel
    f += image(num_rows - 1, 0);
    v += image(num_rows - 1, 0);     // bottom-left vertex
    e += 2 * image(num_rows - 1, 0); // left edge + bottom edge
    v += image(num_rows - 1, 0) || image(num_rows - 2, 0); // top-left vertex
    v +=
        image(num_rows - 1, 0) || image(num_rows - 1, 1); // bottom-right vertex
    v += image(num_rows - 1, 0) || image(num_rows - 2, 0) ||
         image(num_rows - 2, 1) || image(num_rows - 1, 1); // top-right vertex
    e += image(num_rows - 1, 0) || image(num_rows - 2, 0); // top edge
    e += image(num_rows - 1, 0) || image(num_rows - 1, 1); // right edge

    // bottom-right pixel
    f += image(num_rows - 1, num_cols - 1);
    v += image(num_rows - 1, num_cols - 1);     // bottom-right vertex
    e += 2 * image(num_rows - 1, num_cols - 1); // bottom edge + right edge
    e += image(num_rows - 1, num_cols - 1) ||
         image(num_rows - 2, num_cols - 1); // upper edge
    v += image(num_rows - 1, num_cols - 1) ||
         image(num_rows - 2, num_cols - 1); // top-right vertex

    for (size_t i = 1; i < num_rows - 1; ++i) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            f += image(i, j);

            v += image(i, j) || image(i - 1, j) || image(i - 1, j + 1) ||
                 image(i, j + 1);                // top-right vertex
            e += image(i, j) || image(i - 1, j); // top edge
            e += image(i, j) || image(i, j + 1); // right edge
        }
    }

    return v - e + f;
}

// Computing the EC of a binary image according to: S. B. Gray, "Local
// Properties of Binary Images in Two Dimensions," in IEEE Transactions on
// Computers, vol. C-20, no. 5, pp. 551-561, May 1971,
// doi: 10.1109/T-C.1971.223289.
int py_ec_binary_image_2d_gray(py::array_t<uint8_t> image) {
    auto img = image.unchecked<2>();

    size_t num_rows = img.shape(0);
    size_t num_cols = img.shape(1);

    int q1 = 0, q3 = 0, qd = 0;

    // top row
    for (size_t j = 0; j < num_rows - 1; ++j) {
        int a = img(0, j);
        int b = img(0, j + 1);
        q1 += (a && !b) || (!a && b);
    }
    // right row
    for (size_t i = 0; i < num_cols - 1; ++i) {
        int a = img(i, 0);
        int b = img(i + 1, 0);
        q1 += (a && !b) || (!a && b);
    }
    // bottom row
    for (size_t j = 0; j < num_rows - 1; ++j) {
        int a = img(num_rows - 1, j);
        int b = img(num_rows - 1, j + 1);
        q1 += (a && !b) || (!a && b);
    }
    // left row
    for (size_t i = 0; i < num_cols - 1; ++i) {
        int a = img(i, num_cols - 1);
        int b = img(i + 1, num_cols - 1);
        q1 += (a && !b) || (!a && b);
    }
    // top-left bit quad
    q1 += img(0, 0);
    // top-right bit quad
    q1 += img(0, num_cols - 1);
    // bottom-left bit quad
    q1 += img(num_rows - 1, 0);
    // bottom-right bit quad
    q1 += img(num_rows - 1, num_cols - 1);

    for (size_t i = 0; i < num_rows - 1; ++i) {
        for (size_t j = 0; j < num_cols - 1; ++j) {
            uint8_t a = img(i, j);
            uint8_t b = img(i, j + 1);
            uint8_t c = img(i + 1, j);
            uint8_t d = img(i + 1, j + 1);

            uint8_t count = a + b + c + d;

            if (count == 1)
                q1++;
            else if (count == 3)
                q3++;
            else if (count == 2 && a == d)
                qd++;
        }
    }

    return (q1 - q3 - 2 * qd) / 4;
}

Matrix2d<uint8_t> py_binary_threshold_image_2d(py::array_t<int> image,
                                               int val) {
    auto img = image.unchecked<2>();

    size_t num_rows = img.shape(0);
    size_t num_cols = img.shape(1);

    Matrix2d<uint8_t> thresholded_img(num_rows, num_cols, 0);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            thresholded_img(i, j) = img(i, j) <= val;
        }
    }

    return thresholded_img;
}

// BELT(2,3)

bool nth_bit(int number, int n) { return (number >> n) & 1; }

int euler_change_neigh_matrix(int bin_neigh) {
    bool n1 = nth_bit(bin_neigh, 1);
    bool n3 = nth_bit(bin_neigh, 3);
    bool n4 = nth_bit(bin_neigh, 4);
    bool n6 = nth_bit(bin_neigh, 6);
    uint8_t edges = n1 + n4 + n6 + n3;
    bool vA = n3 || nth_bit(bin_neigh, 0) || n1;
    bool vB = n1 || nth_bit(bin_neigh, 2) || n4;
    bool vC = n4 || nth_bit(bin_neigh, 7) || n6;
    bool vD = n6 || nth_bit(bin_neigh, 5) || n3;
    uint8_t vertices = vA + vB + vC + vD;
    return 1 - vertices + edges;
}

std::vector<int> euler_changes_2d() {
    std::vector<int> euler_changes(256, 0);

    for (int i = 0; i < 256; i++) {
        euler_changes[i] = euler_change_neigh_matrix(i);
    }

    return euler_changes;
}

py::array_t<int> py_euler_changes_2d() {
    std::vector<int> euler_changes = euler_changes_2d();

    return py::array_t<int>(256, euler_changes.data());
}

Matrix2d<uint8_t> py_binary_threshold_image_2d3c(py::array_t<int> image_c1,
                                                 py::array_t<int> image_c2,
                                                 py::array_t<int> image_c3,
                                                 int r, int s, int t) {
    auto img_c1 = image_c1.unchecked<2>();
    auto img_c2 = image_c2.unchecked<2>();
    auto img_c3 = image_c3.unchecked<2>();

    size_t num_rows = img_c1.shape(0);
    size_t num_cols = img_c2.shape(1);

    Matrix2d<uint8_t> thresholded_img(num_rows, num_cols, 0);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            thresholded_img(i, j) =
                img_c1(i, j) <= r && img_c2(i, j) <= s && img_c3(i, j) <= t;
        }
    }

    return thresholded_img;
}

py::array_t<int> py_ecp_ind_2d3c_naive(py::array_t<int> image_c1,
                                       py::array_t<int> image_c2,
                                       py::array_t<int> image_c3, int T1,
                                       int T2, int T3) {
    Matrix3d<int> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 1),
                      static_cast<size_t>(T3 + 1), 0);

    for (int r = 0; r <= T1; r++) {
        for (int s = 0; s <= T2; s++) {
            for (int t = 0; t <= T3; t++) {
                Matrix2d<uint8_t> K_rst = py_binary_threshold_image_2d3c(
                    image_c1, image_c2, image_c3, r, s, t);
                ecp(r, s, t) = ec_binary_image_2d_naive(K_rst);
            }
        }
    }

    py::array_t<int> result({ecp.dim(0), ecp.dim(1), ecp.dim(2)});
    auto r = result.mutable_unchecked<3>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                r(v, s, t) = ecp(v, s, t);
            }
        }
    }

    return result;
}

std::bitset<8> neigh_matrix(Matrix2d<int> &image, int i, int j, int f) {
    std::bitset<8> neigh;

    for (size_t p = 0; p < 3; ++p) {
        for (size_t q = 0; q < 3; ++q) {
            if (q + 3 * p < 4) {
                neigh[q + 3 * p] = image(i - 1 + p, j - 1 + q) <= f;
            } else if (q + 3 * p > 4) {
                neigh[q + 3 * p - 1] = image(i - 1 + p, j - 1 + q) <= f;
            }
        }
    }

    return neigh;
}

std::bitset<8> neigh_matrix_lex(Matrix2d<int> &image, int i, int j, int f) {
    std::bitset<8> neigh;

    for (size_t p = 0; p < 3; ++p) {
        for (size_t q = 0; q < 3; ++q) {
            if (q + 3 * p < 4) {
                neigh[q + 3 * p] = image(i - 1 + p, j - 1 + q) <= f;
                // test(p,q) = image(i-1+p,j-1+q) <= f;
            } else if (q + 3 * p > 4) {
                neigh[q + 3 * p - 1] = image(i - 1 + p, j - 1 + q) <= f - 1;
                // test(p,q) = image(i-1+p,j-1+q) <= f-1;
            }
        }
    }

    return neigh;
}

// Matrix2d<int> pad_img_2d(py::array_t<int> image, int val) {
//     auto img = image.unchecked<2>();
//
//     size_t N = img.shape(0);
//     size_t M = img.shape(1);
//
//     Matrix2d<int> padded_img(N + 2, M + 2, val);
//
//     for (size_t i = 1; i < N + 1; ++i) {
//         for (size_t j = 1; j < M + 1; ++j) {
//             padded_img(i, j) = img(i - 1, j - 1);
//         }
//     }
//
//     return padded_img;
// }

Matrix2d<int> pad_img_2d(py::array_t<int, py::array::c_style> image, int val) {
    size_t N = image.shape(0);
    size_t M = image.shape(1);

    // 1. Allocate the padded matrix
    Matrix2d<int> padded_img(N + 2, M + 2, val);

    const int *src_ptr = image.data();
    int *dest_ptr = padded_img.data_ptr();
    size_t padded_cols = M + 2;

    // 2. Copy row by row using highly optimized contiguous memory transfers
    for (size_t i = 0; i < N; ++i) {
        const int *src_row = src_ptr + (i * M);
        int *dest_row = dest_ptr + ((i + 1) * padded_cols) + 1;

        std::copy(src_row, src_row + M, dest_row);
    }

    return padded_img;
}

py::array_t<int> py_ecp_belt_2d3c(py::array_t<int> image_c1,
                                  py::array_t<int> image_c2,
                                  py::array_t<int> image_c3,
                                  py::array_t<int> euler_changes, int T1,
                                  int T2, int T3) {
    auto ec = euler_changes.unchecked<1>();

    size_t N = image_c1.shape(0);
    size_t M = image_c1.shape(1);

    Matrix3d<int> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 1),
                      static_cast<size_t>(T3 + 1), 0);

    Matrix2d<int> padded_img_c1 = pad_img_2d(image_c1, T1 + 1);
    Matrix2d<int> padded_img_c2 = pad_img_2d(image_c2, T2 + 1);
    Matrix2d<int> padded_img_c3 = pad_img_2d(image_c3, T3 + 1);

    for (size_t i = 1; i < N + 1; ++i) {
        for (size_t j = 1; j < M + 1; ++j) {
            int r = padded_img_c1(i, j);
            std::bitset<8> binary_neigh1 =
                neigh_matrix_lex(padded_img_c1, i, j, r);

            std::vector<int> thresholds2(10, T2 + 1); // can be an array
            int s = padded_img_c2(i, j);
            std::vector<int> thresholds3(10, T3 + 1); // can be an array
            int t = padded_img_c3(i, j);
            for (size_t p = 0; p < 3; ++p) {
                for (size_t q = 0; q < 3; ++q) {
                    int val = padded_img_c2(i - 1 + p, j - 1 + q);
                    if (val >= s)
                        thresholds2[q + p * 3] = val;
                    val = padded_img_c3(i - 1 + p, j - 1 + q);
                    if (val >= t)
                        thresholds3[q + p * 3] = val;
                }
            }
            sort(thresholds2.begin(), thresholds2.end());
            auto last = unique(thresholds2.begin(), thresholds2.end());
            thresholds2.erase(last, thresholds2.end());

            sort(thresholds3.begin(), thresholds3.end());
            last = unique(thresholds3.begin(), thresholds3.end());
            thresholds3.erase(last, thresholds3.end());

            for (size_t z = 0; z < thresholds2.size() - 1; ++z) {
                size_t start2 = static_cast<size_t>(thresholds2[z]);
                size_t end2 = static_cast<size_t>(thresholds2[z + 1]);

                std::bitset<8> binary_neigh2 =
                    neigh_matrix(padded_img_c2, i, j, thresholds2[z]);

                int last_change = 0;
                for (size_t y = 0; y < thresholds3.size() - 1; ++y) {
                    size_t start3 = static_cast<size_t>(thresholds3[y]);

                    std::bitset<8> binary_neigh3 =
                        neigh_matrix(padded_img_c3, i, j, thresholds3[y]);
                    std::bitset<8> binary_neigh =
                        binary_neigh1 & binary_neigh2 & binary_neigh3;

                    size_t neigh_num = binary_neigh.to_ulong();
                    int change = ec(neigh_num);

                    ecp(static_cast<int>(r), start2, start3) +=
                        change - last_change;
                    if (end2 <= T2)
                        ecp(static_cast<int>(r), end2, start3) -=
                            change - last_change;
                    last_change = change;
                }
            }
        }
    }

    for (int r = 0; r <= T1; ++r) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 1; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s, t - 1);
            }
        }
    }
    for (int r = 0; r <= T1; ++r) {
        for (int s = 1; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s - 1, t);
            }
        }
    }

    for (size_t r = 1; r < T1 + 1; ++r) {
        for (size_t s = 0; s < T2 + 1; ++s) {
            for (size_t t = 0; t < T3 + 1; ++t) {
                ecp(r, s, t) += ecp(r - 1, s, t);
            }
        }
    }

    /*
        auto capsule = py::capsule(ecp, [](void *f) {
            delete reinterpret_cast<Matrix<int> *>(f);
        });
    */

    py::array_t<int> result({(size_t)T1 + 1, (size_t)T2 + 1, (size_t)T3 + 1});
    auto r = result.mutable_unchecked<3>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                r(v, s, t) = ecp(v, s, t);
            }
        }
    }

    return result;
}

// HEWA(2,3)

py::array_t<int> py_ecp_hewa_2d3c(py::array_t<int> image_c1,
                                  py::array_t<int> image_c2,
                                  py::array_t<int> image_c3, int T1, int T2,
                                  int T3) {
    size_t N = image_c1.shape(0);
    size_t M = image_c1.shape(1);

    Matrix3d<int> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 1),
                      static_cast<size_t>(T3 + 1), 0);

    Matrix2d<int> padded_img_c1 = pad_img_2d(image_c1, T1 + 1);
    Matrix2d<int> padded_img_c2 = pad_img_2d(image_c2, T2 + 1);
    Matrix2d<int> padded_img_c3 = pad_img_2d(image_c3, T3 + 1);

    for (size_t i = 1; i < N + 1; ++i) {
        for (size_t j = 1; j < M + 1; ++j) {
            int r = padded_img_c1(i, j);

            std::vector<int> thresholds2(10, T2 + 1); // can be an array
            int s = padded_img_c2(i, j);
            std::vector<int> thresholds3(10, T3 + 1); // can be an array
            int t = padded_img_c3(i, j);
            for (size_t p = 0; p < 3; ++p) {
                for (size_t q = 0; q < 3; ++q) {
                    int val = padded_img_c2(i - 1 + p, j - 1 + q);
                    if (val >= s)
                        thresholds2[q + p * 3] = val;
                    val = padded_img_c3(i - 1 + p, j - 1 + q);
                    if (val >= t)
                        thresholds3[q + p * 3] = val;
                }
            }
            sort(thresholds2.begin(), thresholds2.end());
            auto last = unique(thresholds2.begin(), thresholds2.end());
            thresholds2.erase(last, thresholds2.end());

            sort(thresholds3.begin(), thresholds3.end());
            last = unique(thresholds3.begin(), thresholds3.end());
            thresholds3.erase(last, thresholds3.end());

            for (size_t z = 0; z < thresholds2.size() - 1; ++z) {
                size_t start2 = static_cast<size_t>(thresholds2[z]);
                size_t end2 = static_cast<size_t>(thresholds2[z + 1]);

                int last_change = 0;
                for (size_t y = 0; y < thresholds3.size() - 1; ++y) {
                    size_t start3 = static_cast<size_t>(thresholds3[y]);

                    auto is_active = [&](int di, int dj) {
                        return padded_img_c2(i + di, j + dj) <= start2 &&
                               padded_img_c3(i + di, j + dj) <= start3;
                    };

                    auto get_n = [&](int di, int dj) {
                        return is_active(di, dj) ? padded_img_c1(i + di, j + dj)
                                                 : T1 + 1;
                    };

                    // Standard Euler calculation
                    int n1 = get_n(-1, -1), n2 = get_n(-1, 0),
                        n3 = get_n(-1, 1);
                    int n4 = get_n(0, -1), n6 = get_n(0, 1);
                    int n7 = get_n(1, -1), n8 = get_n(1, 0), n9 = get_n(1, 1);

                    int change = 1;
                    // edges
                    change -= (n2 >= r) + (n6 > r) + (n8 > r) + (n4 >= r);
                    // vertices
                    change += (n1 >= r && n2 >= r && n4 >= r);
                    change += (n2 >= r && n3 >= r && n6 > r);
                    change += (n4 >= r && n7 > r && n8 > r);
                    change += (n6 > r && n8 > r && n9 > r);

                    // Apply 2D Difference markers to the (S, T) slice at this V
                    int delta = change - last_change;
                    ecp(r, start2, start3) += delta;
                    if (end2 <= T2)
                        ecp(r, end2, start3) -= delta;
                    last_change = change;
                }
            }
        }
    }

    for (int r = 0; r <= T1; ++r) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 1; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s, t - 1);
            }
        }
    }
    for (int r = 0; r <= T1; ++r) {
        for (int s = 1; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s - 1, t);
            }
        }
    }

    for (size_t r = 1; r < T1 + 1; ++r) {
        for (size_t s = 0; s < T2 + 1; ++s) {
            for (size_t t = 0; t < T3 + 1; ++t) {
                ecp(r, s, t) += ecp(r - 1, s, t);
            }
        }
    }

    /*
        auto capsule = py::capsule(ecp, [](void *f) {
            delete reinterpret_cast<Matrix<int> *>(f);
        });
    */

    py::array_t<int> result({(size_t)T1 + 1, (size_t)T2 + 1, (size_t)T3 + 1});
    auto r = result.mutable_unchecked<3>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                r(v, s, t) = ecp(v, s, t);
            }
        }
    }

    return result;
}

// HALE:T(2,3)

py::array_t<int>
py_ecp_hale_t_2d3c(py::array_t<int, py::array::c_style> image_c1,
                   py::array_t<int, py::array::c_style> image_c2,
                   py::array_t<int, py::array::c_style> image_c3, int T1,
                   int T2, int T3) {
    Matrix2d<int> img_c1(image_c1);
    Matrix2d<int> img_c2(image_c2);
    Matrix2d<int> img_c3(image_c3);

    size_t num_rows = image_c1.shape(0);
    size_t num_cols = image_c2.shape(1);

    Matrix3d<int> ecp(T1 + 1, T2 + 1, T3 + 1, 0);

    // o----
    // |
    // |

    // inner pixels
    for (size_t i = 1; i < num_rows - 1; ++i) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            // top-left vertex
            ecp(std::min({img_c1(i, j), img_c1(i - 1, j), img_c1(i - 1, j - 1),
                          img_c1(i, j - 1)}),
                std::min({img_c2(i, j), img_c2(i - 1, j), img_c2(i - 1, j - 1),
                          img_c2(i, j - 1)}),
                std::min({img_c3(i, j), img_c3(i - 1, j), img_c3(i - 1, j - 1),
                          img_c3(i, j - 1)})) += 1;

            // top edge
            ecp(std::min({img_c1(i, j), img_c1(i - 1, j)}),
                std::min({img_c2(i, j), img_c2(i - 1, j)}),
                std::min({img_c3(i, j), img_c3(i - 1, j)})) += -1;

            // left edge
            ecp(std::min({img_c1(i, j), img_c1(i, j - 1)}),
                std::min({img_c2(i, j), img_c2(i, j - 1)}),
                std::min({img_c3(i, j), img_c3(i, j - 1)})) += -1;

            // square
            ecp(img_c1(i, j), img_c2(i, j), img_c3(i, j)) += 1;
        }
    }

    // bottom-edge pixels
    if (num_rows >= 2) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            // top-left vertex
            ecp(std::min({img_c1(num_rows - 1, j), img_c1(num_rows - 2, j),
                          img_c1(num_rows - 2, j - 1),
                          img_c1(num_rows - 1, j - 1)}),
                std::min({img_c2(num_rows - 1, j), img_c2(num_rows - 2, j),
                          img_c2(num_rows - 2, j - 1),
                          img_c2(num_rows - 1, j - 1)}),
                std::min({img_c3(num_rows - 1, j), img_c3(num_rows - 2, j),
                          img_c3(num_rows - 2, j - 1),
                          img_c3(num_rows - 1, j - 1)})) += 1;

            // bottom-left vertex - is canceled by the left edge

            // top edge
            ecp(std::min({img_c1(num_rows - 1, j), img_c1(num_rows - 2, j)}),
                std::min({img_c2(num_rows - 1, j), img_c2(num_rows - 2, j)}),
                std::min({img_c3(num_rows - 1, j), img_c3(num_rows - 2, j)})) +=
                -1;

            // left edge - is canceled by the bottom-right vertex
            // bottom edge - is canceled by the square
            // square - is canceled by the bottom edge
        }
    } else {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            // top-left vertex
            ecp(std::min(
                    {img_c1(num_rows - 1, j), img_c1(num_rows - 1, j - 1)}),
                std::min(
                    {img_c2(num_rows - 1, j), img_c2(num_rows - 1, j - 1)}),
                std::min({img_c3(num_rows - 1, j),
                          img_c3(num_rows - 1, j - 1)})) += 1;

            // bottom-left vertex - is canceled by the left edge

            // top edge
            ecp(img_c1(num_rows - 1, j), img_c2(num_rows - 1, j),
                img_c3(num_rows - 1, j)) += -1;

            // left edge - is canceled by the bottom-right vertex
            // bottom edge - is canceled by the square
            // square - is canceled by the bottom edge
        }
    }

    // right-edge pixels
    if (num_cols >= 2) {
        for (size_t i = 1; i < num_rows - 1; ++i) {
            // top-left vertex
            ecp(std::min({img_c1(i, num_cols - 1), img_c1(i - 1, num_cols - 1),
                          img_c1(i - 1, num_cols - 2),
                          img_c1(i, num_cols - 2)}),
                std::min({img_c2(i, num_cols - 1), img_c2(i - 1, num_cols - 1),
                          img_c2(i - 1, num_cols - 2),
                          img_c2(i, num_cols - 2)}),
                std::min({img_c3(i, num_cols - 1), img_c3(i - 1, num_cols - 1),
                          img_c3(i - 1, num_cols - 2),
                          img_c3(i, num_cols - 2)})) += 1;

            // top-right vertex - is canceled by the top edge

            // top edge - is canceled by the top-right vertex

            // left edge
            ecp(std::min({img_c1(i, num_cols - 1), img_c1(i, num_cols - 2)}),
                std::min({img_c2(i, num_cols - 1), img_c2(i, num_cols - 2)}),
                std::min({img_c3(i, num_cols - 1), img_c3(i, num_cols - 2)})) +=
                -1;

            // right edge - is canceled by the square

            // square - is canceled by the left edge
        }
    } else {
        for (size_t i = 1; i < num_rows - 1; ++i) {
            // top-left vertex
            ecp(std::min(
                    {img_c1(i, num_cols - 1), img_c1(i - 1, num_cols - 1)}),
                std::min(
                    {img_c2(i, num_cols - 1), img_c2(i - 1, num_cols - 1)}),
                std::min({img_c3(i, num_cols - 1),
                          img_c3(i - 1, num_cols - 1)})) += 1;

            // top-right vertex - is canceled by the top edge

            // top edge - is canceled by the top-right vertex

            // left edge
            ecp(img_c1(i, num_cols - 1), img_c2(i, num_cols - 1),
                img_c3(i, num_cols - 1)) += -1;

            // right edge - is canceled by the square

            // square - is canceled by the left edge
        }
    }

    // top-left pixel
    // top-right vertex - is canceled by the right edge
    // top-left vertex - is canceled by the top edge
    // top edge - is canceled by the top-left vertex
    // right edge - is canceled by the top-right vertex
    // left edge - is canceled by the square
    // square - is canceled by the left edge

    // top-right pixel
    // top-right vertex - is canceled by the top edge
    // top edge - is canceled by the top-right vertex
    // right edge - is canceled by the square
    // square - is canceled by the right edge

    // bottom-right pixel
    // top-right vertex - is canceled by the top edge
    // bottom-right vertex - is canceled by the right edge
    //
    // top edge - is canceled by the top-right vertex
    // right edge - is canceled by the bottom-right vertex
    // bottom edge - is canceled by the square
    //
    // square - is canceled by the bottom edge

    // bottom-right pixel
    // top-left vertex
    if (num_rows >= 2) {
        if (num_cols >= 2) {
            ecp(std::min({img_c1(num_rows - 1, num_cols - 1),
                          img_c1(num_rows - 2, num_cols - 1),
                          img_c1(num_rows - 2, num_cols - 2),
                          img_c1(num_rows - 1, num_cols - 2)}),
                std::min({img_c2(num_rows - 1, num_cols - 1),
                          img_c2(num_rows - 2, num_cols - 1),
                          img_c2(num_rows - 2, num_cols - 2),
                          img_c2(num_rows - 1, num_cols - 2)}),
                std::min({img_c3(num_rows - 1, num_cols - 1),
                          img_c3(num_rows - 2, num_cols - 1),
                          img_c3(num_rows - 2, num_cols - 2),
                          img_c3(num_rows - 1, num_cols - 2)})) += 1;
        } else {
            ecp(std::min({img_c1(num_rows - 1, num_cols - 1),
                          img_c1(num_rows - 2, num_cols - 1)}),
                std::min({img_c2(num_rows - 1, num_cols - 1),
                          img_c2(num_rows - 2, num_cols - 1)}),
                std::min({img_c3(num_rows - 1, num_cols - 1),
                          img_c3(num_rows - 2, num_cols - 1)})) += 1;
        }
    } else {
        if (num_cols >= 2) {
            ecp(std::min({img_c1(num_rows - 1, num_cols - 1),
                          img_c1(num_rows - 1, num_cols - 2)}),
                std::min({img_c2(num_rows - 1, num_cols - 1),
                          img_c2(num_rows - 1, num_cols - 2)}),
                std::min({img_c3(num_rows - 1, num_cols - 1),
                          img_c3(num_rows - 1, num_cols - 2)})) += 1;
        } else {
            ecp(img_c1(num_rows - 1, num_cols - 1),
                img_c2(num_rows - 1, num_cols - 1),
                img_c3(num_rows - 1, num_cols - 1)) += 1;
        }
    }
    // top-right vertex - is canceled by the top edge
    // bottom-right vertex - is canceled by the right edge
    // bottom-left vertex - is canceled by the left edge
    //
    // top edge - is canceled by the top-left vertex
    // right edge - is canceled by the bottom-right vertex
    // bottom edge - is canceled by the square
    // left edge - is canceled by the bottom-left vertex
    //
    // square - is canceled by the bottom edge

    for (int r = 0; r <= T1; ++r) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 1; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s, t - 1);
            }
        }
    }
    for (int r = 0; r <= T1; ++r) {
        for (int s = 1; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s - 1, t);
            }
        }
    }

    for (size_t r = 1; r < T1 + 1; ++r) {
        for (size_t s = 0; s < T2 + 1; ++s) {
            for (size_t t = 0; t < T3 + 1; ++t) {
                ecp(r, s, t) += ecp(r - 1, s, t);
            }
        }
    }

    py::array_t<int> result({(size_t)T1 + 1, (size_t)T2 + 1, (size_t)T3 + 1});
    auto r = result.mutable_unchecked<3>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                r(v, s, t) = ecp(v, s, t);
            }
        }
    }

    return result;
}

// HALE:V(2,3)

py::array_t<int>
py_ecp_hale_v_2d3c(py::array_t<int, py::array::c_style> image_c1,
                   py::array_t<int, py::array::c_style> image_c2,
                   py::array_t<int, py::array::c_style> image_c3, int T1,
                   int T2, int T3) {
    Matrix2d<int> img_c1(image_c1);
    Matrix2d<int> img_c2(image_c2);
    Matrix2d<int> img_c3(image_c3);

    size_t num_rows = image_c1.shape(0);
    size_t num_cols = image_c1.shape(1);

    Matrix3d<int> ecp(T1 + 1, T2 + 1, T3 + 1, 0);

    //     |
    //     |
    // ----o

    // inner pixels
    for (size_t i = 1; i < num_rows; ++i) {
        for (size_t j = 1; j < num_cols; ++j) {
            // square
            ecp(std::max({img_c1(i, j), img_c1(i - 1, j), img_c1(i - 1, j - 1),
                          img_c1(i, j - 1)}),
                std::max({img_c2(i, j), img_c2(i - 1, j), img_c2(i - 1, j - 1),
                          img_c2(i, j - 1)}),
                std::max({img_c3(i, j), img_c3(i - 1, j), img_c3(i - 1, j - 1),
                          img_c3(i, j - 1)})) += 1;

            // bottom edge
            ecp(std::max({img_c1(i, j), img_c1(i, j - 1)}),
                std::max({img_c2(i, j), img_c2(i, j - 1)}),
                std::max({img_c3(i, j), img_c3(i, j - 1)})) += -1;

            // right edge
            ecp(std::max({img_c1(i, j), img_c1(i - 1, j)}),
                std::max({img_c2(i, j), img_c2(i - 1, j)}),
                std::max({img_c3(i, j), img_c3(i - 1, j)})) += -1;

            // bottom-right vertex
            ecp(img_c1(i, j), img_c2(i, j), img_c3(i, j)) += 1;
        }
    }

    // pixels at the left edge
    for (size_t i = 1; i < num_rows; ++i) {
        // bottom-right vertex
        ecp(img_c1(i, 0), img_c2(i, 0), img_c3(i, 0)) += 1;

        // right edge
        ecp(std::max({img_c1(i, 0), img_c1(i - 1, 0)}),
            std::max({img_c2(i, 0), img_c2(i - 1, 0)}),
            std::max({img_c3(i, 0), img_c3(i - 1, 0)})) += -1;
    }

    // pixels at the top edge
    for (size_t j = 1; j < num_cols; ++j) {
        // top-left vertex
        ecp(img_c1(0, j), img_c2(0, j), img_c3(0, j)) += 1;

        // top edge
        ecp(std::max({img_c1(0, j), img_c1(0, j - 1)}),
            std::max({img_c2(0, j), img_c2(0, j - 1)}),
            std::max({img_c3(0, j), img_c3(0, j - 1)})) += -1;
    }

    // top-left pixel
    ecp(img_c1(0, 0), img_c2(0, 0), img_c3(0, 0)) += 1;

    for (int r = 0; r <= T1; ++r) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 1; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s, t - 1);
            }
        }
    }
    for (int r = 0; r <= T1; ++r) {
        for (int s = 1; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s - 1, t);
            }
        }
    }

    for (size_t r = 1; r < T1 + 1; ++r) {
        for (size_t s = 0; s < T2 + 1; ++s) {
            for (size_t t = 0; t < T3 + 1; ++t) {
                ecp(r, s, t) += ecp(r - 1, s, t);
            }
        }
    }

    py::array_t<int> result({(size_t)T1 + 1, (size_t)T2 + 1, (size_t)T3 + 1});
    auto r = result.mutable_unchecked<3>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                r(v, s, t) = ecp(v, s, t);
            }
        }
    }

    return result;
}

// HALE:I(2,3)

int pwr_sgn(int n) { return 1 - 2 * (1 & n); }

void place_markers(Matrix3d<int> &ecp, std::initializer_list<int> channel1,
                   std::initializer_list<int> channel2,
                   std::initializer_list<int> channel3, size_t dim) {
    size_t set_size = channel1.size();
    size_t num_subsets = 1 << set_size;

    for (size_t mask = 1; mask < num_subsets; ++mask) {
        int current_max_c1 = std::numeric_limits<int>::min();
        int current_max_c2 = std::numeric_limits<int>::min();
        int current_max_c3 = std::numeric_limits<int>::min();

        for (size_t j = 0; j < set_size; ++j) {
            if (mask & (1 << j)) {
                current_max_c1 =
                    std::max(current_max_c1, *std::next(channel1.begin(), j));
                current_max_c2 =
                    std::max(current_max_c2, *std::next(channel2.begin(), j));
                current_max_c3 =
                    std::max(current_max_c3, *std::next(channel3.begin(), j));
            }
        }

        int subset_size = __builtin_popcountll(mask);

        ecp(current_max_c1, current_max_c2, current_max_c3) +=
            pwr_sgn(dim + subset_size - 1);
    }
}

py::array_t<int>
py_ecp_hale_i_2d3c(py::array_t<int, py::array::c_style> image_c1,
                   py::array_t<int, py::array::c_style> image_c2,
                   py::array_t<int, py::array::c_style> image_c3, int T1,
                   int T2, int T3) {
    Matrix2d<int> img_c1(image_c1);
    Matrix2d<int> img_c2(image_c2);
    Matrix2d<int> img_c3(image_c3);

    size_t num_rows = image_c1.shape(0);
    size_t num_cols = image_c1.shape(1);

    Matrix3d<int> ecp(T1 + 1, T2 + 1, T3 + 1, 0);

    // o----
    // |
    // |

    // inner pixels
    for (size_t i = 1; i < num_rows - 1; ++i) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            // top-left vertex
            place_markers(ecp,
                          {img_c1(i, j), img_c1(i - 1, j), img_c1(i - 1, j - 1),
                           img_c1(i, j - 1)},
                          {img_c2(i, j), img_c2(i - 1, j), img_c2(i - 1, j - 1),
                           img_c2(i, j - 1)},
                          {img_c3(i, j), img_c3(i - 1, j), img_c3(i - 1, j - 1),
                           img_c3(i, j - 1)},
                          0);

            // top edge
            place_markers(ecp, {img_c1(i, j), img_c1(i - 1, j)},
                          {img_c2(i, j), img_c2(i - 1, j)},
                          {img_c3(i, j), img_c3(i - 1, j)}, 1);

            // left edge
            place_markers(ecp, {img_c1(i, j), img_c1(i, j - 1)},
                          {img_c2(i, j), img_c2(i, j - 1)},
                          {img_c3(i, j), img_c3(i, j - 1)}, 1);

            // square
            ecp(img_c1(i, j), img_c2(i, j), img_c3(i, j)) += 1;
        }
    }

    // bottom-edge pixels
    if (num_rows >= 2) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            // top-left vertex
            place_markers(
                ecp,
                {img_c1(num_rows - 1, j), img_c1(num_rows - 2, j),
                 img_c1(num_rows - 2, j - 1), img_c1(num_rows - 1, j - 1)},
                {img_c2(num_rows - 1, j), img_c2(num_rows - 2, j),
                 img_c2(num_rows - 2, j - 1), img_c2(num_rows - 1, j - 1)},
                {img_c3(num_rows - 1, j), img_c3(num_rows - 2, j),
                 img_c3(num_rows - 2, j - 1), img_c3(num_rows - 1, j - 1)},
                0);

            // bottom-left vertex - is canceled by the left edge

            // top edge
            place_markers(
                ecp, {img_c1(num_rows - 1, j), img_c1(num_rows - 2, j)},
                {img_c2(num_rows - 1, j), img_c2(num_rows - 2, j)},
                {img_c3(num_rows - 1, j), img_c3(num_rows - 2, j)}, 1);

            // left edge - is canceled by the bottom-right vertex
            // bottom edge - is canceled by the square
            // square - is canceled by the bottom edge
        }
    } else {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            // top-left vertex
            place_markers(
                ecp, {img_c1(num_rows - 1, j), img_c1(num_rows - 1, j - 1)},
                {img_c2(num_rows - 1, j), img_c2(num_rows - 1, j - 1)},
                {img_c3(num_rows - 1, j), img_c3(num_rows - 1, j - 1)}, 0);

            // bottom-left vertex - is canceled by the left edge

            // top edge
            place_markers(ecp, {img_c1(num_rows - 1, j)},
                          {img_c2(num_rows - 1, j)}, {img_c3(num_rows - 1, j)},
                          1);

            // left edge - is canceled by the bottom-right vertex
            // bottom edge - is canceled by the square
            // square - is canceled by the bottom edge
        }
    }

    // right-edge pixels
    if (num_cols >= 2) {
        for (size_t i = 1; i < num_rows - 1; ++i) {
            // top-left vertex
            place_markers(
                ecp,
                {img_c1(i, num_cols - 1), img_c1(i - 1, num_cols - 1),
                 img_c1(i - 1, num_cols - 2), img_c1(i, num_cols - 2)},
                {img_c2(i, num_cols - 1), img_c2(i - 1, num_cols - 1),
                 img_c2(i - 1, num_cols - 2), img_c2(i, num_cols - 2)},
                {img_c3(i, num_cols - 1), img_c3(i - 1, num_cols - 1),
                 img_c3(i - 1, num_cols - 2), img_c3(i, num_cols - 2)},
                0);

            // top-right vertex - is canceled by the top edge

            // top edge - is canceled by the top-right vertex

            // left edge
            place_markers(
                ecp, {img_c1(i, num_cols - 1), img_c1(i, num_cols - 2)},
                {img_c2(i, num_cols - 1), img_c2(i, num_cols - 2)},
                {img_c3(i, num_cols - 1), img_c3(i, num_cols - 2)}, 1);

            // right edge - is canceled by the square

            // square - is canceled by the left edge
        }
    } else {
        for (size_t i = 1; i < num_rows - 1; ++i) {
            // top-left vertex
            place_markers(
                ecp, {img_c1(i, num_cols - 1), img_c1(i - 1, num_cols - 1)},
                {img_c2(i, num_cols - 1), img_c2(i - 1, num_cols - 1)},
                {img_c3(i, num_cols - 1), img_c3(i - 1, num_cols - 1)}, 0);

            // top-right vertex - is canceled by the top edge

            // top edge - is canceled by the top-right vertex

            // left edge
            place_markers(ecp, {img_c1(i, num_cols - 1)},
                          {img_c2(i, num_cols - 1)}, {img_c3(i, num_cols - 1)},
                          1);

            // right edge - is canceled by the square

            // square - is canceled by the left edge
        }
    }

    // top-left pixel
    // top-right vertex - is canceled by the right edge
    // top-left vertex - is canceled by the top edge
    // top edge - is canceled by the top-left vertex
    // right edge - is canceled by the top-right vertex
    // left edge - is canceled by the square
    // square - is canceled by the left edge

    // top-right pixel
    // top-right vertex - is canceled by the top edge
    // top edge - is canceled by the top-right vertex
    // right edge - is canceled by the square
    // square - is canceled by the right edge

    // bottom-right pixel
    // top-right vertex - is canceled by the top edge
    // bottom-right vertex - is canceled by the right edge
    //
    // top edge - is canceled by the top-right vertex
    // right edge - is canceled by the bottom-right vertex
    // bottom edge - is canceled by the square
    //
    // square - is canceled by the bottom edge

    // bottom-right pixel
    // top-left vertex
    if (num_rows >= 2) {
        if (num_cols >= 2) {
            place_markers(ecp,
                          {img_c1(num_rows - 1, num_cols - 1),
                           img_c1(num_rows - 2, num_cols - 1),
                           img_c1(num_rows - 2, num_cols - 2),
                           img_c1(num_rows - 1, num_cols - 2)},
                          {img_c2(num_rows - 1, num_cols - 1),
                           img_c2(num_rows - 2, num_cols - 1),
                           img_c2(num_rows - 2, num_cols - 2),
                           img_c2(num_rows - 1, num_cols - 2)},
                          {img_c3(num_rows - 1, num_cols - 1),
                           img_c3(num_rows - 2, num_cols - 1),
                           img_c3(num_rows - 2, num_cols - 2),
                           img_c3(num_rows - 1, num_cols - 2)},
                          0);
        } else {
            place_markers(ecp,
                          {img_c1(num_rows - 1, num_cols - 1),
                           img_c1(num_rows - 2, num_cols - 1)},
                          {img_c2(num_rows - 1, num_cols - 1),
                           img_c2(num_rows - 2, num_cols - 1)},
                          {img_c3(num_rows - 1, num_cols - 1),
                           img_c3(num_rows - 2, num_cols - 1)},
                          0);
        }
    } else {
        if (num_cols >= 2) {
            place_markers(ecp,
                          {img_c1(num_rows - 1, num_cols - 1),
                           img_c1(num_rows - 1, num_cols - 2)},
                          {img_c2(num_rows - 1, num_cols - 1),
                           img_c2(num_rows - 1, num_cols - 2)},
                          {img_c3(num_rows - 1, num_cols - 1),
                           img_c3(num_rows - 1, num_cols - 2)},
                          0);

        } else {
            ecp(img_c1(num_rows - 1, num_cols - 1),
                img_c2(num_rows - 1, num_cols - 1),
                img_c2(num_rows - 1, num_cols - 1)) += 1;
        }
    }
    // top-right vertex - is canceled by the top edge
    // bottom-right vertex - is canceled by the right edge
    // bottom-left vertex - is canceled by the left edge
    //
    // top edge - is canceled by the top-left vertex
    // right edge - is canceled by the bottom-right vertex
    // bottom edge - is canceled by the square
    // left edge - is canceled by the bottom-left vertex
    //
    // square - is canceled by the bottom edge

    for (int r = 0; r <= T1; ++r) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 1; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s, t - 1);
            }
        }
    }
    for (int r = 0; r <= T1; ++r) {
        for (int s = 1; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s - 1, t);
            }
        }
    }

    for (size_t r = 1; r < T1 + 1; ++r) {
        for (size_t s = 0; s < T2 + 1; ++s) {
            for (size_t t = 0; t < T3 + 1; ++t) {
                ecp(r, s, t) += ecp(r - 1, s, t);
            }
        }
    }

    py::array_t<int> result({(size_t)T1 + 1, (size_t)T2 + 1, (size_t)T3 + 1});
    auto r = result.mutable_unchecked<3>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                r(v, s, t) = ecp(v, s, t);
            }
        }
    }

    return result;
}

py::array_t<int>
list_of_minimal_grades_top_cell(py::array_t<int, py::array::c_style> image_c1,
                                py::array_t<int, py::array::c_style> image_c2,
                                py::array_t<int, py::array::c_style> image_c3) {
    Matrix2d<int> img_c1(image_c1);
    Matrix2d<int> img_c2(image_c2);
    Matrix2d<int> img_c3(image_c3);

    size_t num_rows = image_c1.shape(0);
    size_t num_cols = image_c2.shape(1);

    // Matrix:
    // [ dimension-sign, R, G, B,
    // dimension-sign, R, G, B, ... ]
    size_t num_cubes = (2 * num_rows + 1) * (2 * num_cols + 1);
    Matrix2d<int> contributions(num_cubes, 4, 0);

    size_t cube_counter = 0;

    // INNER PIXELS
    for (size_t i = 1; i < num_rows - 1; ++i) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            // top-right vertex
            contributions(cube_counter, 0) = 1;
            contributions(cube_counter, 1) =
                std::min({img_c1(i, j), img_c1(i - 1, j), img_c1(i - 1, j + 1),
                          img_c1(i, j + 1)});
            contributions(cube_counter, 2) =
                std::min({img_c2(i, j), img_c2(i - 1, j), img_c2(i - 1, j + 1),
                          img_c2(i, j + 1)});
            contributions(cube_counter, 3) =
                std::min({img_c3(i, j), img_c3(i - 1, j), img_c3(i - 1, j + 1),
                          img_c3(i, j + 1)});
            cube_counter++;

            // top edge
            contributions(cube_counter, 0) = -1;
            contributions(cube_counter, 1) =
                std::min({img_c1(i, j), img_c1(i - 1, j)});
            contributions(cube_counter, 2) =
                std::min({img_c2(i, j), img_c2(i - 1, j)});
            contributions(cube_counter, 3) =
                std::min({img_c3(i, j), img_c3(i - 1, j)});
            cube_counter++;

            // right edge
            contributions(cube_counter, 0) = -1;
            contributions(cube_counter, 1) =
                std::min({img_c1(i, j), img_c1(i, j + 1)});
            contributions(cube_counter, 2) =
                std::min({img_c2(i, j), img_c2(i, j + 1)});
            contributions(cube_counter, 3) =
                std::min({img_c3(i, j), img_c3(i, j + 1)});
            cube_counter++;

            // square
            contributions(cube_counter, 0) = 1;
            contributions(cube_counter, 1) = img_c1(i, j);
            contributions(cube_counter, 2) = img_c2(i, j);
            contributions(cube_counter, 3) = img_c3(i, j);
            cube_counter++;
        }
    }

    // TOP PIXELS
    for (size_t j = 1; j < num_cols - 1; ++j) {
        // top-right vertex - is canceled by the right edge
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) =
            std::min({img_c1(0, j), img_c1(0, j + 1)});
        contributions(cube_counter, 2) =
            std::min({img_c2(0, j), img_c2(0, j + 1)});
        contributions(cube_counter, 3) =
            std::min({img_c3(0, j), img_c3(0, j + 1)});
        cube_counter++;

        // top edge - is canceled by the square
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) = img_c1(0, j);
        contributions(cube_counter, 2) = img_c2(0, j);
        contributions(cube_counter, 3) = img_c3(0, j);
        cube_counter++;

        // right edge - is canceled by the top-right vertex
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) =
            std::min({img_c1(0, j), img_c1(0, j + 1)});
        contributions(cube_counter, 2) =
            std::min({img_c2(0, j), img_c2(0, j + 1)});
        contributions(cube_counter, 3) =
            std::min({img_c3(0, j), img_c3(0, j + 1)});
        cube_counter++;

        // square - is canceled by the top edge
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) = img_c1(0, j);
        contributions(cube_counter, 2) = img_c2(0, j);
        contributions(cube_counter, 3) = img_c3(0, j);
        cube_counter++;
    }

    // RIGHT PIXELS
    for (size_t i = 1; i < num_rows - 1; ++i) {
        // top-right vertex - is canceled by the top edge
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) =
            std::min({img_c1(i, num_cols - 1), img_c1(i - 1, num_cols - 1)});
        contributions(cube_counter, 2) =
            std::min({img_c2(i, num_cols - 1), img_c2(i - 1, num_cols - 1)});
        contributions(cube_counter, 3) =
            std::min({img_c3(i, num_cols - 1), img_c3(i - 1, num_cols - 1)});
        cube_counter++;

        // top edge - is canceled by the top-right vertex
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) =
            std::min({img_c1(i, num_cols - 1), img_c1(i - 1, num_cols - 1)});
        contributions(cube_counter, 2) =
            std::min({img_c2(i, num_cols - 1), img_c2(i - 1, num_cols - 1)});
        contributions(cube_counter, 3) =
            std::min({img_c3(i, num_cols - 1), img_c3(i - 1, num_cols - 1)});
        cube_counter++;

        // right edge - is canceled by the square
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) = img_c1(i, num_cols - 1);
        contributions(cube_counter, 2) = img_c2(i, num_cols - 1);
        contributions(cube_counter, 3) = img_c3(i, num_cols - 1);
        cube_counter++;

        // square - is canceled by the right edge
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) = img_c1(i, num_cols - 1);
        contributions(cube_counter, 2) = img_c2(i, num_cols - 1);
        contributions(cube_counter, 3) = img_c3(i, num_cols - 1);
        cube_counter++;
    }

    // BOTTOM PIXELS
    for (size_t j = 1; j < num_cols - 1; ++j) {
        // top-right vertex
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) = std::min(
            {img_c1(num_rows - 1, j), img_c1(num_rows - 2, j),
             img_c1(num_rows - 2, j + 1), img_c1(num_rows - 1, j + 1)});
        contributions(cube_counter, 2) = std::min(
            {img_c2(num_rows - 1, j), img_c2(num_rows - 2, j),
             img_c2(num_rows - 2, j + 1), img_c2(num_rows - 1, j + 1)});
        contributions(cube_counter, 3) = std::min(
            {img_c3(num_rows - 1, j), img_c3(num_rows - 2, j),
             img_c3(num_rows - 2, j + 1), img_c3(num_rows - 1, j + 1)});
        cube_counter++;

        // bottom-right vertex - is canceled by the right edge
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) =
            std::min({img_c1(num_rows - 1, j), img_c1(num_rows - 1, j + 1)});
        contributions(cube_counter, 2) =
            std::min({img_c2(num_rows - 1, j), img_c2(num_rows - 1, j + 1)});
        contributions(cube_counter, 3) =
            std::min({img_c3(num_rows - 1, j), img_c3(num_rows - 1, j + 1)});
        cube_counter++;

        // top edge
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) =
            std::min({img_c1(num_rows - 1, j), img_c1(num_rows - 2, j)});
        contributions(cube_counter, 2) =
            std::min({img_c2(num_rows - 1, j), img_c2(num_rows - 2, j)});
        contributions(cube_counter, 3) =
            std::min({img_c3(num_rows - 1, j), img_c3(num_rows - 2, j)});
        cube_counter++;

        // right edge - is canceled by the bottom-right vertex
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) =
            std::min({img_c1(num_rows - 1, j), img_c1(num_rows - 1, j + 1)});
        contributions(cube_counter, 2) =
            std::min({img_c2(num_rows - 1, j), img_c2(num_rows - 1, j + 1)});
        contributions(cube_counter, 3) =
            std::min({img_c3(num_rows - 1, j), img_c3(num_rows - 1, j + 1)});
        cube_counter++;

        // bottom edge - is canceled by the square
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) = img_c1(num_rows - 1, j);
        contributions(cube_counter, 2) = img_c2(num_rows - 1, j);
        contributions(cube_counter, 3) = img_c3(num_rows - 1, j);
        cube_counter++;

        // square - is canceled by the bottom edge
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) = img_c1(num_rows - 1, j);
        contributions(cube_counter, 2) = img_c2(num_rows - 1, j);
        contributions(cube_counter, 3) = img_c3(num_rows - 1, j);
        cube_counter++;
    }

    // LEFT PIXELS
    for (size_t i = 1; i < num_rows - 1; ++i) {
        // top-right vertex
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) = std::min(
            {img_c1(i, 0), img_c1(i - 1, 0), img_c1(i - 1, 1), img_c1(i, 1)});
        contributions(cube_counter, 2) = std::min(
            {img_c2(i, 0), img_c2(i - 1, 0), img_c2(i - 1, 1), img_c2(i, 1)});
        contributions(cube_counter, 3) = std::min(
            {img_c3(i, 0), img_c3(i - 1, 0), img_c3(i - 1, 1), img_c3(i, 1)});
        cube_counter++;

        // top-left vertex - is canceled by the top edge
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) =
            std::min({img_c1(i, 0), img_c1(i - 1, 0)});
        contributions(cube_counter, 2) =
            std::min({img_c2(i, 0), img_c2(i - 1, 0)});
        contributions(cube_counter, 3) =
            std::min({img_c3(i, 0), img_c3(i - 1, 0)});
        cube_counter++;

        // top edge - is canceled by the top-left vertex
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) =
            std::min({img_c1(i, 0), img_c1(i - 1, 0)});
        contributions(cube_counter, 2) =
            std::min({img_c2(i, 0), img_c2(i - 1, 0)});
        contributions(cube_counter, 3) =
            std::min({img_c3(i, 0), img_c3(i - 1, 0)});
        cube_counter++;

        // right edge
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) = std::min({img_c1(i, 0), img_c1(i, 1)});
        contributions(cube_counter, 2) = std::min({img_c2(i, 0), img_c2(i, 1)});
        contributions(cube_counter, 3) = std::min({img_c3(i, 0), img_c3(i, 1)});
        cube_counter++;

        // left edge - is canceled by the square
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) = img_c1(i, 0);
        contributions(cube_counter, 2) = img_c2(i, 0);
        contributions(cube_counter, 3) = img_c3(i, 0);
        cube_counter++;

        // square - is canceled by the left edge
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) = img_c1(i, 0);
        contributions(cube_counter, 2) = img_c2(i, 0);
        contributions(cube_counter, 3) = img_c3(i, 0);
        cube_counter++;
    }

    // TOP-LEFT PIXEL
    // top-right vertex - is canceled by the right edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = std::min({img_c1(0, 0), img_c1(0, 1)});
    contributions(cube_counter, 2) = std::min({img_c2(0, 0), img_c2(0, 1)});
    contributions(cube_counter, 3) = std::min({img_c3(0, 0), img_c3(0, 1)});
    cube_counter++;

    // top-left vertex - is canceled by the top edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = img_c1(0, 0);
    contributions(cube_counter, 2) = img_c2(0, 0);
    contributions(cube_counter, 3) = img_c3(0, 0);
    cube_counter++;

    // top edge - is canceled by the top-left vertex
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) = img_c1(0, 0);
    contributions(cube_counter, 2) = img_c2(0, 0);
    contributions(cube_counter, 3) = img_c3(0, 0);
    cube_counter++;

    // right edge - is canceled by the top-right vertex
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) = std::min({img_c1(0, 0), img_c1(0, 1)});
    contributions(cube_counter, 2) = std::min({img_c2(0, 0), img_c2(0, 1)});
    contributions(cube_counter, 3) = std::min({img_c3(0, 0), img_c3(0, 1)});
    cube_counter++;

    // left edge - is canceled by the square
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) = img_c1(0, 0);
    contributions(cube_counter, 2) = img_c2(0, 0);
    contributions(cube_counter, 3) = img_c3(0, 0);
    cube_counter++;

    // square - is canceled by the left edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = img_c1(0, 0);
    contributions(cube_counter, 2) = img_c2(0, 0);
    contributions(cube_counter, 3) = img_c3(0, 0);
    cube_counter++;

    // TOP-RIGHT PIXEL
    // top-right vertex - is canceled by the top edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = img_c1(0, num_cols - 1);
    contributions(cube_counter, 2) = img_c2(0, num_cols - 1);
    contributions(cube_counter, 3) = img_c3(0, num_cols - 1);
    cube_counter++;

    // top edge - is canceled by the top-right vertex
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) = img_c1(0, num_cols - 1);
    contributions(cube_counter, 2) = img_c2(0, num_cols - 1);
    contributions(cube_counter, 3) = img_c3(0, num_cols - 1);
    cube_counter++;

    // right edge - is canceled by the square
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) = img_c1(0, num_cols - 1);
    contributions(cube_counter, 2) = img_c2(0, num_cols - 1);
    contributions(cube_counter, 3) = img_c3(0, num_cols - 1);
    cube_counter++;

    // square - is canceled by the right edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = img_c1(0, num_cols - 1);
    contributions(cube_counter, 2) = img_c2(0, num_cols - 1);
    contributions(cube_counter, 3) = img_c3(0, num_cols - 1);
    cube_counter++;

    // BOTTOM-RIGHT PIXEL
    // top-right vertex - is canceled by the top edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) =
        std::min({img_c1(num_rows - 1, num_cols - 1),
                  img_c1(num_rows - 2, num_cols - 1)});
    contributions(cube_counter, 2) =
        std::min({img_c2(num_rows - 1, num_cols - 1),
                  img_c2(num_rows - 2, num_cols - 1)});
    contributions(cube_counter, 3) =
        std::min({img_c3(num_rows - 1, num_cols - 1),
                  img_c3(num_rows - 2, num_cols - 1)});
    cube_counter++;

    // bottom-right vertex - is canceled by the right edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = img_c1(num_rows - 1, num_cols - 1);
    contributions(cube_counter, 2) = img_c2(num_rows - 1, num_cols - 1);
    contributions(cube_counter, 3) = img_c3(num_rows - 1, num_cols - 1);
    cube_counter++;

    // top edge - is canceled by the top-right vertex
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) =
        std::min({img_c1(num_rows - 1, num_cols - 1),
                  img_c1(num_rows - 2, num_cols - 1)});
    contributions(cube_counter, 2) =
        std::min({img_c2(num_rows - 1, num_cols - 1),
                  img_c2(num_rows - 2, num_cols - 1)});
    contributions(cube_counter, 3) =
        std::min({img_c3(num_rows - 1, num_cols - 1),
                  img_c3(num_rows - 2, num_cols - 1)});
    cube_counter++;

    // right edge - is canceled by the bottom-right vertex
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) = img_c1(num_rows - 1, num_cols - 1);
    contributions(cube_counter, 2) = img_c2(num_rows - 1, num_cols - 1);
    contributions(cube_counter, 3) = img_c3(num_rows - 1, num_cols - 1);
    cube_counter++;

    // bottom edge - is canceled by the square
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) = img_c1(num_rows - 1, num_cols - 1);
    contributions(cube_counter, 2) = img_c2(num_rows - 1, num_cols - 1);
    contributions(cube_counter, 3) = img_c3(num_rows - 1, num_cols - 1);
    cube_counter++;

    // square - is canceled by the bottom edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = img_c1(num_rows - 1, num_cols - 1);
    contributions(cube_counter, 2) = img_c2(num_rows - 1, num_cols - 1);
    contributions(cube_counter, 3) = img_c3(num_rows - 1, num_cols - 1);
    cube_counter++;

    // BOTTOM-LEFT PIXEL
    // top-right vertex
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) =
        std::min({img_c1(num_rows - 1, 0), img_c1(num_rows - 2, 0),
                  img_c1(num_rows - 2, 1), img_c1(num_rows - 1, 1)});
    contributions(cube_counter, 2) =
        std::min({img_c2(num_rows - 1, 0), img_c2(num_rows - 2, 0),
                  img_c2(num_rows - 2, 1), img_c2(num_rows - 1, 1)});
    contributions(cube_counter, 3) =
        std::min({img_c3(num_rows - 1, 0), img_c3(num_rows - 2, 0),
                  img_c3(num_rows - 2, 1), img_c3(num_rows - 1, 1)});
    cube_counter++;

    // top-left vertex - is canceled by the top edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) =
        std::min({img_c1(num_rows - 1, 0), img_c1(num_rows - 2, 0)});
    contributions(cube_counter, 2) =
        std::min({img_c2(num_rows - 1, 0), img_c2(num_rows - 2, 0)});
    contributions(cube_counter, 3) =
        std::min({img_c3(num_rows - 1, 0), img_c3(num_rows - 2, 0)});
    cube_counter++;

    // bottom-right vertex - is canceled by the right edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) =
        std::min({img_c1(num_rows - 1, 0), img_c1(num_rows - 1, 1)});
    contributions(cube_counter, 2) =
        std::min({img_c2(num_rows - 1, 0), img_c2(num_rows - 1, 1)});
    contributions(cube_counter, 3) =
        std::min({img_c3(num_rows - 1, 0), img_c3(num_rows - 1, 1)});
    cube_counter++;

    // bottom-left vertex - is canceled by the left edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = img_c1(num_rows - 1, 0);
    contributions(cube_counter, 2) = img_c2(num_rows - 1, 0);
    contributions(cube_counter, 3) = img_c3(num_rows - 1, 0);
    cube_counter++;

    // top edge - is canceled by the top-left vertex
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) =
        std::min({img_c1(num_rows - 1, 0), img_c1(num_rows - 2, 0)});
    contributions(cube_counter, 2) =
        std::min({img_c2(num_rows - 1, 0), img_c2(num_rows - 2, 0)});
    contributions(cube_counter, 3) =
        std::min({img_c3(num_rows - 1, 0), img_c3(num_rows - 2, 0)});
    cube_counter++;

    // right edge - is canceled by the bottom-right vertex
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) =
        std::min({img_c1(num_rows - 1, 0), img_c1(num_rows - 1, 1)});
    contributions(cube_counter, 2) =
        std::min({img_c2(num_rows - 1, 0), img_c2(num_rows - 1, 1)});
    contributions(cube_counter, 3) =
        std::min({img_c3(num_rows - 1, 0), img_c3(num_rows - 1, 1)});
    cube_counter++;

    // bottom edge - is canceled by the square
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) = img_c1(num_rows - 1, 0);
    contributions(cube_counter, 2) = img_c2(num_rows - 1, 0);
    contributions(cube_counter, 3) = img_c3(num_rows - 1, 0);
    cube_counter++;

    // left edge - is canceled by the bottom-left vertex
    contributions(cube_counter, 0) = -1;
    contributions(cube_counter, 1) = img_c1(num_rows - 1, 0);
    contributions(cube_counter, 2) = img_c2(num_rows - 1, 0);
    contributions(cube_counter, 3) = img_c3(num_rows - 1, 0);
    cube_counter++;

    // square - is canceled by the bottom edge
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = img_c1(num_rows - 1, 0);
    contributions(cube_counter, 2) = img_c2(num_rows - 1, 0);
    contributions(cube_counter, 3) = img_c3(num_rows - 1, 0);
    cube_counter++;

    py::array_t<int> result({num_cubes, (size_t)4});
    auto r = result.mutable_unchecked<2>();
    for (int i = 0; i < num_cubes; ++i) {
        r(i, 0) = contributions(i, 0);
        r(i, 1) = contributions(i, 1);
        r(i, 2) = contributions(i, 2);
        r(i, 3) = contributions(i, 3);
    }

    return result;
}

py::array_t<int>
py_ecp_hale_eulearning_2d3c(py::array_t<int, py::array::c_style> contributions,
                            int T1, int T2, int T3) {
    Matrix2d<int> contribs(contributions);
    size_t num_cubes = contributions.shape(0);

    Matrix3d<int> ecp(T1 + 1, T2 + 1, T3 + 1, 0);

    for (size_t i = 0; i < num_cubes; ++i) {
        ecp(contribs(i, 1), contribs(i, 2), contribs(i, 3)) += contribs(i, 0);
    }

    for (int r = 0; r <= T1; ++r) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 1; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s, t - 1);
            }
        }
    }
    for (int r = 0; r <= T1; ++r) {
        for (int s = 1; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r, s - 1, t);
            }
        }
    }

    for (size_t r = 1; r <= T1; ++r) {
        for (size_t s = 0; s <= T2; ++s) {
            for (size_t t = 0; t <= T3; ++t) {
                ecp(r, s, t) += ecp(r - 1, s, t);
            }
        }
    }

    py::array_t<int> result({(size_t)T1 + 1, (size_t)T2 + 1, (size_t)T3 + 1});
    auto r = result.mutable_unchecked<3>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                r(v, s, t) = ecp(v, s, t);
            }
        }
    }

    return result;
}

py::array_t<int>
list_of_minimal_grades_vertex(py::array_t<int, py::array::c_style> image_c1,
                              py::array_t<int, py::array::c_style> image_c2,
                              py::array_t<int, py::array::c_style> image_c3) {
    Matrix2d<int> img_c1(image_c1);
    Matrix2d<int> img_c2(image_c2);
    Matrix2d<int> img_c3(image_c3);

    size_t num_rows = image_c1.shape(0);
    size_t num_cols = image_c2.shape(1);

    size_t num_cubes = (2 * num_rows - 1) * (2 * num_cols - 1);
    Matrix2d<int> contributions(num_cubes, 4, 0);

    size_t cube_counter = 0;

    //     |
    //     |
    // ----o

    // inner pixels
    for (size_t i = 1; i < num_rows; ++i) {
        for (size_t j = 1; j < num_cols; ++j) {
            // square
            contributions(cube_counter, 0) = 1;
            contributions(cube_counter, 1) =
                std::max({img_c1(i, j), img_c1(i - 1, j), img_c1(i - 1, j - 1),
                          img_c1(i, j - 1)});
            contributions(cube_counter, 2) =
                std::max({img_c2(i, j), img_c2(i - 1, j), img_c2(i - 1, j - 1),
                          img_c2(i, j - 1)});
            contributions(cube_counter, 3) =
                std::max({img_c3(i, j), img_c3(i - 1, j), img_c3(i - 1, j - 1),
                          img_c3(i, j - 1)});
            cube_counter++;

            // bottom edge
            contributions(cube_counter, 0) = -1;
            contributions(cube_counter, 1) =
                std::max({img_c1(i, j), img_c1(i, j - 1)});
            contributions(cube_counter, 2) =
                std::max({img_c2(i, j), img_c2(i, j - 1)});
            contributions(cube_counter, 3) =
                std::max({img_c3(i, j), img_c3(i, j - 1)});
            cube_counter++;

            // right edge
            contributions(cube_counter, 0) = -1;
            contributions(cube_counter, 1) =
                std::max({img_c1(i, j), img_c1(i - 1, j)});
            contributions(cube_counter, 2) =
                std::max({img_c2(i, j), img_c2(i - 1, j)});
            contributions(cube_counter, 3) =
                std::max({img_c3(i, j), img_c3(i - 1, j)});
            cube_counter++;

            // bottom-right vertex
            contributions(cube_counter, 0) = 1;
            contributions(cube_counter, 1) = img_c1(i, j);
            contributions(cube_counter, 2) = img_c2(i, j);
            contributions(cube_counter, 3) = img_c3(i, j);
            cube_counter++;
        }
    }

    // pixels at the left edge
    for (size_t i = 1; i < num_rows; ++i) {
        // bottom-right vertex
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) = img_c1(i, 0);
        contributions(cube_counter, 2) = img_c2(i, 0);
        contributions(cube_counter, 3) = img_c3(i, 0);
        cube_counter++;

        // right edge
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) =
            std::max({img_c1(i, 0), img_c1(i - 1, 0)});
        contributions(cube_counter, 2) =
            std::max({img_c2(i, 0), img_c2(i - 1, 0)});
        contributions(cube_counter, 3) =
            std::max({img_c3(i, 0), img_c3(i - 1, 0)});
        cube_counter++;
    }

    // pixels at the top edge
    for (size_t j = 1; j < num_cols; ++j) {
        // top-left vertex
        contributions(cube_counter, 0) = 1;
        contributions(cube_counter, 1) = img_c1(0, j);
        contributions(cube_counter, 2) = img_c2(0, j);
        contributions(cube_counter, 3) = img_c3(0, j);
        cube_counter++;

        // top edge
        contributions(cube_counter, 0) = -1;
        contributions(cube_counter, 1) =
            std::max({img_c1(0, j), img_c1(0, j - 1)});
        contributions(cube_counter, 2) =
            std::max({img_c2(0, j), img_c2(0, j - 1)});
        contributions(cube_counter, 3) =
            std::max({img_c3(0, j), img_c3(0, j - 1)});
        cube_counter++;
    }

    // top-left pixel
    contributions(cube_counter, 0) = 1;
    contributions(cube_counter, 1) = img_c1(0, 0);
    contributions(cube_counter, 2) = img_c2(0, 0);
    contributions(cube_counter, 3) = img_c3(0, 0);

    py::array_t<int> result({num_cubes, (size_t)4});
    auto r = result.mutable_unchecked<2>();
    for (int i = 0; i < num_cubes; ++i) {
        r(i, 0) = contributions(i, 0);
        r(i, 1) = contributions(i, 1);
        r(i, 2) = contributions(i, 2);
        r(i, 3) = contributions(i, 3);
    }

    return result;
}

PYBIND11_MODULE(_core, m) {
    m.def("belt_2d_euler_changes", &py_euler_changes_2d,
          "Precomputed the local changes used for the BELT(2,3) algorithm");
    m.def("ecp_ind_2d3c_naive", &py_ecp_ind_2d3c_naive,
          "Naive algorithm to compute the ECP of the filtration induced by "
          "top-cells of a 2-dimensional 3-channel image using the EC of "
          "thresholded binary images");
    m.def("ecp_belt_2d3c", &py_ecp_belt_2d3c,
          "BELT(2,3) algorithm to compute the ECP of the filtration induced by "
          "top-cells of a 2-dimensional 3-channel image");
    m.def("ecp_hewa_2d3c", &py_ecp_hewa_2d3c,
          "HEWA(2,3) algorithm to compute the ECP of the filtration induced by "
          "top-cells of a 2-dimensional 3-channel image");
    m.def("ecp_hale_t_2d3c", &py_ecp_hale_t_2d3c,
          "HALE:T(2,3) algorithm to compute the ECP of the top-cell filtration "
          "of a 2-dimensional 3-channel image");
    m.def("ecp_hale_v_2d3c", &py_ecp_hale_v_2d3c,
          "HALE:V(2,3) algorithm to compute the ECP of the vertex filtration "
          "of a 2-dimensional 3-channel image");
    m.def(
        "ecp_hale_i_2d3c", &py_ecp_hale_i_2d3c,
        "HALE:I(2,3) algorithm to compute the ECP of the filtration induced by "
        "top-cells of a 2-dimensional 3-channel image");

    m.def(
        "list_of_minimal_grades_top_cell", &list_of_minimal_grades_top_cell,
        "Yields list of the form [dimension-sign, t1, t2, t3, dimension-sign, "
        "t1, t2, t3, ...] containing the minimal grade (t1,t2,t3) = f(sigma) "
        "for every elementary cube with respect to the top-cell filtration");
    m.def("ecp_hale_eulearning_2d3c", &py_ecp_hale_eulearning_2d3c,
          "Computes the ECP of a sublevel filtration based on a precomputed "
          "list of minimal grades");
    m.def(
        "list_of_minimal_grades_vertex", &list_of_minimal_grades_vertex,
        "Yields list of the form [dimension-sign, t1, t2, t3, dimension-sign, "
        "t1, t2, t3, ...] containing the minimal grade (t1,t2,t3) = f(sigma) "
        "for every elementary cube with respect to the vertex filtration");
}
