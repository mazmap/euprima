#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include <array>
#include <bitset>
#include <stdexcept>
#include <vector>

#include <algorithm> // for std::min

#include <iomanip>
#include <iostream>

template <typename T, size_t N> class Matrix {
  private:
    std::vector<T> data;
    std::vector<size_t> shape;
    std::array<size_t, N> strides; // possible optimization by using std::array
                                   // instead, no just use native arrays lol

    size_t idx(const std::vector<size_t> &indices) const {
        size_t idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            idx += indices[i] * strides[i];
        }
        return idx;
    }

  public:
    Matrix(std::initializer_list<size_t> dims, T init_val = T())
        : shape(dims), strides() {

        size_t total_size = 1;
        for (auto d : shape)
            total_size *= d;
        data.assign(total_size, init_val);

        // Strides for row-major order
        size_t current_stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }
    }

    template <typename... Args> T &operator()(Args... args) {
        static_assert(sizeof...(Args) > 0,
                      "Matrix requires at least one index");

        size_t idx = 0;
        size_t dim = 0;

        // "comma-fold" suggested by Gemini
        ([&] { idx += static_cast<size_t>(args) * strides[dim++]; }(), ...);

        return data[idx];
    }

    template <typename... Args> const T &operator()(Args... args) const {
        static_assert(sizeof...(Args) > 0,
                      "Matrix requires at least one index");

        size_t idx = 0;
        size_t dim = 0;

        // "comma-fold" suggested by Gemini
        ([&] { idx += static_cast<size_t>(args) * strides[dim++]; }(), ...);

        return data[idx];
    }

    T &operator()(size_t r, size_t c) {
        return data[r * strides[0] + c * strides[1]];
    }

    size_t dim(size_t index) { return shape[index]; }

    const size_t dim(size_t index) const { return shape[index]; }

    T *data_ptr() { return data.data(); }
};

template <typename T> class Matrix<T, 2> {
    size_t rows, cols;
    std::vector<T> data;

  public:
    Matrix(size_t r, size_t c, T init_val = T())
        : rows(r), cols(c), data(r * c, init_val) {}

    // No folds, no lambdas. Pure, simple math the compiler loves.
    T &operator()(size_t r, size_t c) { return data[r * cols + c]; }

    const T &operator()(size_t r, size_t c) const { return data[r * cols + c]; }

    size_t dim(size_t index) { return index == 0 ? rows : cols; }

    const size_t dim(size_t index) const { return index == 0 ? rows : cols; }

    T *data_ptr() { return data.data(); }
};

template <typename T> class Matrix<T, 3> {
    size_t dim1, dim2, dim3;
    std::vector<T> data;

  public:
    Matrix(size_t d1, size_t d2, size_t d3, T init_val = T())
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

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os, const Matrix<T, N> &mat) {
    for (size_t i = 0; i < mat.dim(0); ++i) {
        os << "[ ";
        for (size_t j = 0; j < mat.dim(1); ++j) {
            // Using setw(4) keeps columns aligned even with different digit
            // counts
            os << std::setw(4) << mat(i, j) << " ";
        }
        os << "]\n";
    }
    return os;
}

template <size_t N> std::bitset<N> reverse(std::bitset<N> b) {
    std::bitset<N> res;
    for (size_t i = 0; i < N; ++i) {
        if (b[i]) {
            res.set(N - 1 - i);
        }
    }
    return res;
}

// bool vs uint_8!
int ec_binary_image_2d_naive(Matrix<uint8_t, 2> &image) {
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

int py_ec_binary_image_2d_naive(py::array_t<uint8_t> image) {
    auto img = image.unchecked<2>();

    size_t num_rows = img.shape(0);
    size_t num_cols = img.shape(1);

    int v = 0;
    int e = 0;
    int f = 0;

    for (size_t j = 1; j < num_cols - 1; ++j) {
        // iterate through pixels at the top edge
        f += img(0, j);
        v += img(0, j) || img(0, j + 1); // top-right vertex
        e += img(0, j);                  // upper edge
        e += img(0, j) || img(0, j + 1); // right edge
    }

    for (size_t i = 1; i < num_rows - 1; ++i) {
        // iterate through pixels at right edge
        f += img(i, num_cols - 1);
        v += img(i, num_cols - 1) ||
             img(i - 1, num_cols - 1); // top-right vertex
        e += img(i, num_cols - 1) || img(i - 1, num_cols - 1); // upper edge
        e += img(i, num_cols - 1);                             // right edge
    }

    for (size_t i = 1; i < num_rows - 1; ++i) {
        // iterate through pixels at left edge
        f += img(i, 0);

        v += img(i, 0) || img(i - 1, 0); // top-left vertex
        e += img(i, 0);                  // left edge

        v += img(i, 0) || img(i - 1, 0) || img(i - 1, 1) ||
             img(i, 1);                  // upper right vertex
        e += img(i, 0) || img(i - 1, 0); // upper edge
        e += img(i, 0) || img(i, 1);     // right edge
    }

    for (size_t j = 1; j < num_cols - 1; ++j) {
        // iterate through pixels at the bottom edge
        f += img(num_rows - 1, j);

        v += img(num_rows - 1, j) ||
             img(num_rows - 1, j + 1); // bottom-right vertex
        e += img(num_rows - 1, j);     // bottom edge

        v += img(num_rows - 1, j) || img(num_rows - 2, j) ||
             img(num_rows - 2, j + 1) ||
             img(num_rows - 1, j + 1);                     // upper right vertex
        e += img(num_rows - 1, j) || img(num_rows - 2, j); // upper edge
        e += img(num_rows - 1, j) || img(num_rows - 1, j + 1); // right edge
    }

    // top-left pixel
    f += img(0, 0);
    v += img(0, 0) || img(0, 1); // top-right vertex
    e += 2 * img(0, 0);          // upper edge + left edge
    e += img(0, 0) || img(0, 1); // right edge
    v += img(0, 0);              // top-left vertex

    // top-right pixel
    f += img(0, num_cols - 1);
    v += img(0, num_cols - 1);     // top-right vertex
    e += 2 * img(0, num_cols - 1); // top edge + right edge

    // bottom-left pixel
    f += img(num_rows - 1, 0);
    v += img(num_rows - 1, 0);     // bottom-left vertex
    e += 2 * img(num_rows - 1, 0); // left edge + bottom edge
    v += img(num_rows - 1, 0) || img(num_rows - 2, 0); // top-left vertex
    v += img(num_rows - 1, 0) || img(num_rows - 1, 1); // bottom-right vertex
    v += img(num_rows - 1, 0) || img(num_rows - 2, 0) || img(num_rows - 2, 1) ||
         img(num_rows - 1, 1);                         // top-right vertex
    e += img(num_rows - 1, 0) || img(num_rows - 2, 0); // top edge
    e += img(num_rows - 1, 0) || img(num_rows - 1, 1); // right edge

    // bottom-right pixel
    f += img(num_rows - 1, num_cols - 1);
    v += img(num_rows - 1, num_cols - 1);     // bottom-right vertex
    e += 2 * img(num_rows - 1, num_cols - 1); // bottom edge + right edge
    e += img(num_rows - 1, num_cols - 1) ||
         img(num_rows - 2, num_cols - 1); // upper edge
    v += img(num_rows - 1, num_cols - 1) ||
         img(num_rows - 2, num_cols - 1); // top-right vertex

    for (size_t i = 1; i < num_rows - 1; ++i) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            f += img(i, j);

            v += img(i, j) || img(i - 1, j) || img(i - 1, j + 1) ||
                 img(i, j + 1);              // top-right vertex
            e += img(i, j) || img(i - 1, j); // top edge
            e += img(i, j) || img(i, j + 1); // right edge
        }
    }

    return v - e + f;
}

int sum_bool_2d(Matrix<uint8_t, 2> &matrix) {
    size_t numI = matrix.dim(0);
    size_t numJ = matrix.dim(1);

    int total = 0;
    for (size_t i = 0; i < numI; ++i) {
        for (size_t j = 0; j < numJ; ++j) {
            if (matrix(i, j) == 1)
                total += 1;
        }
    }

    return total;
}

int sum_bool_2d(py::array_t<uint8_t> matrix) {
    auto img = matrix.unchecked<2>();

    size_t numI = matrix.shape(0);
    size_t numJ = matrix.shape(1);

    int total = 0;
    for (size_t i = 0; i < numI; ++i) {
        for (size_t j = 0; j < numJ; ++j) {
            if (img(i, j) == 1)
                total += 1;
        }
    }

    return total;
}

int char_binary_image_2d(py::array_t<uint8_t> input) {
    auto img = input.unchecked<2>();

    // Input shape: binary image number of rows and columns
    size_t numI = img.shape(0);
    size_t numJ = img.shape(1);

    // Matrices for vectices, horizontal edges and vertical edges
    Matrix<uint8_t, 2> V(numI + 1, numJ + 1, 0);
    Matrix<uint8_t, 2> Eh(numI + 1, numJ, 0);
    Matrix<uint8_t, 2> Ev(numI, numJ + 1, 0);

    // Loop over pixels to update V, Eh, Ev
    for (size_t i = 0; i < numI; ++i) {
        for (size_t j = 0; j < numJ; ++j) {
            if (img(i, j) == 1) {
                V(i, j) = 1;
                V(i + 1, j) = 1;
                V(i, j + 1) = 1;
                V(i + 1, j + 1) = 1;

                Eh(i, j) = 1;
                Eh(i + 1, j) = 1;

                Ev(i, j) = 1;
                Ev(i, j + 1) = 1;
            }
        }
    }

    // Sum of elements in the matrices
    int v = sum_bool_2d(V);
    int eh = sum_bool_2d(Eh);
    int ev = sum_bool_2d(Ev);
    int f = sum_bool_2d(input);

    int EC = v - eh - ev + f;
    return EC;
}

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

// does not work...
int py_ec_binary_image_2d_yao(py::array_t<uint8_t> image) {
    auto uimg = image.unchecked<2>();

    size_t num_rows = uimg.shape(0);
    size_t num_cols = uimg.shape(1);

    Matrix<uint8_t, 2> img(num_rows + 2, num_cols + 2, 0);
    for (size_t i = 1; i < num_rows + 1; ++i) {
        for (size_t j = 1; j < num_cols + 1; ++j) {
            img(i, j) = uimg(i - 1, j - 1);
        }
    }

    num_rows += 2;
    num_cols += 2;

    int w2 = 0, wc = 0;
    size_t x = 2, y = 2;

    while (y <= num_cols) {
        while (x <= num_rows) {
            if (img(x - 2, y - 2) == 1)
                x++;
            else if (img(x - 2, y - 1) == 0) {
            labelA:
                if (img(x - 1, y - 2) == 1) {
                    x += 2;
                } else if (img(x - 1, y - 1) == 0) {
                    x++;
                    if (x <= num_cols)
                        goto labelA;
                } else {
                    w2++;
                    x++;
                    if (x <= num_cols)
                        goto labelB;
                }
            } else {
            labelB:
                if (img(x - 1, y - 2) == 1) {
                    wc++;
                    x += 2;
                } else if (img(x - 1, y - 1) == 0) {
                    x++;
                    if (x <= num_cols)
                        goto labelA;
                } else {
                    x++;
                    if (x <= num_cols)
                        goto labelB;
                }
            }
        }
        y++;
    }

    return w2 - wc;
}

Matrix<uint8_t, 2> py_binary_threshold_image_2d(py::array_t<int> image,
                                                int val) {
    auto img = image.unchecked<2>();

    size_t num_rows = img.shape(0);
    size_t num_cols = img.shape(1);

    Matrix<uint8_t, 2> thresholded_img(num_rows, num_cols, 0);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            thresholded_img(i, j) = img(i, j) <= val;
        }
    }

    return thresholded_img;
}

// ATTENTION: The following specification is not safe at all as Args accepts ANY
// type, not just matrices!
template <typename... Args>
Matrix<uint8_t, 2> elementwise_AND_2d(const Matrix<uint8_t, 2> &image1,
                                      const Args &...images) {
    Matrix<uint8_t, 2> result = image1;

    size_t num_rows = image1.dim(0);
    size_t num_cols = image1.dim(1);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            result(i, j) = (result(i, j) && ... && images(i, j));
        }
    }

    return result;
}

Matrix<uint8_t, 2> elementwise_AND_2d(const Matrix<uint8_t, 2> &image1,
                                      const Matrix<uint8_t, 2> &image2) {
    Matrix<uint8_t, 2> result = image1;

    size_t num_rows = image1.dim(0);
    size_t num_cols = image1.dim(1);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            result(i, j) = (result(i, j) && image2(i, j));
        }
    }

    return result;
}

Matrix<uint8_t, 2> elementwise_AND_2d(const Matrix<uint8_t, 2> &image1,
                                      const Matrix<uint8_t, 2> &image2,
                                      const Matrix<uint8_t, 2> &image3) {
    Matrix<uint8_t, 2> result = image1;

    size_t num_rows = image1.dim(0);
    size_t num_cols = image1.dim(1);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            result(i, j) = result(i, j) && image2(i, j) && image3(i, j);
        }
    }

    return result;
}

py::array_t<int> py_ecp_2d2c_naive(py::array_t<int> image_c1,
                                   py::array_t<int> image_c2, int T1, int T2) {
    Matrix<int, 2> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 1),
                       0);

    for (int i = 0; i <= T1; i++) {
        Matrix<uint8_t, 2> thresholded_img_c1 =
            py_binary_threshold_image_2d(image_c1, i);

        for (int j = 0; j <= T2; j++) {
            Matrix<uint8_t, 2> thresholded_img_c2 =
                py_binary_threshold_image_2d(image_c2, j);
            Matrix<uint8_t, 2> Kij =
                elementwise_AND_2d(thresholded_img_c1, thresholded_img_c2);
            ecp(i, j) = ec_binary_image_2d_naive(Kij);
        }
    }

    return py::array_t<int>({ecp.dim(0), ecp.dim(1)}, ecp.data_ptr());
}

std::bitset<8> neigh_matrix(Matrix<int, 2> &image, int i, int j, int f) {
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

std::bitset<8> neigh_matrix_lex(Matrix<int, 2> &image, int i, int j, int f) {
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

Matrix<int, 2> pad_img_2d(py::array_t<int> image, int val) {
    auto img = image.unchecked<2>();

    size_t N = img.shape(0);
    size_t M = img.shape(1);

    Matrix<int, 2> padded_img(N + 2, M + 2, val);

    for (size_t i = 1; i < N + 1; ++i) {
        for (size_t j = 1; j < M + 1; ++j) {
            padded_img(i, j) = img(i - 1, j - 1);
        }
    }

    return padded_img;
}

py::array_t<int> ecp_2d2c(py::array_t<int> image_c1, py::array_t<int> image_c2,
                          py::array_t<int> euler_changes, int T1, int T2) {
    auto ec = euler_changes.unchecked<1>();

    size_t N = image_c1.shape(0);
    size_t M = image_c1.shape(1);

    Matrix<int, 2> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 1),
                       0);

    Matrix<int, 2> padded_img_c1 = pad_img_2d(image_c1, T1 + 1);
    Matrix<int, 2> padded_img_c2 = pad_img_2d(image_c2, T2 + 1);

    for (size_t i = 1; i < N + 1; ++i) {
        for (size_t j = 1; j < M + 1; ++j) {
            int r = padded_img_c1(i, j);
            std::bitset<8> binary_neigh1 =
                neigh_matrix_lex(padded_img_c1, i, j, r);

            std::vector<int> thresholds2(10, T2 + 1); // can be an array
            int s = padded_img_c2(i, j);
            for (size_t p = 0; p < 3; ++p) {
                for (size_t q = 0; q < 3; ++q) {
                    int val = padded_img_c2(i - 1 + p, j - 1 + q);
                    if (val >= s)
                        thresholds2[q + p * 3] = val;
                }
            }

            sort(thresholds2.begin(), thresholds2.end());
            auto last = unique(thresholds2.begin(), thresholds2.end());
            thresholds2.erase(last, thresholds2.end());

            for (size_t z = 0; z < thresholds2.size() - 1; ++z) {
                size_t start = static_cast<size_t>(thresholds2[z]);
                size_t end = static_cast<size_t>(thresholds2[z + 1]);

                std::bitset<8> binary_neigh2 =
                    neigh_matrix(padded_img_c2, i, j, thresholds2[z]);
                std::bitset<8> binary_neigh = binary_neigh1 & binary_neigh2;

                size_t neigh_num = binary_neigh.to_ulong();
                int change = ec(neigh_num);

                for (size_t ind = start; ind < end; ++ind) {
                    ecp(static_cast<int>(r), ind) += change;
                }
            }
        }
    }

    for (size_t s = 1; s < T1 + 1; ++s) {
        for (size_t t = 0; t < T2 + 1; ++t) {
            ecp(s, t) += ecp(s - 1, t);
        }
    }

    /*
        auto capsule = py::capsule(ecp, [](void *f) {
            delete reinterpret_cast<Matrix<int> *>(f);
        });
    */

    return py::array_t<int>({ecp.dim(0), ecp.dim(1)}, // Shape
                            ecp.data_ptr());
}

py::array_t<int> py_ecp_2d2c_optimized(py::array_t<int> image_c1,
                                       py::array_t<int> image_c2,
                                       py::array_t<int> euler_changes, int T1,
                                       int T2) {
    auto ec = euler_changes.unchecked<1>();

    size_t N = image_c1.shape(0);
    size_t M = image_c1.shape(1);

    Matrix<int, 2> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 1),
                       0);

    Matrix<int, 2> padded_img_c1 = pad_img_2d(image_c1, T1 + 1);
    Matrix<int, 2> padded_img_c2 = pad_img_2d(image_c2, T2 + 1);

    for (size_t i = 1; i < N + 1; ++i) {
        for (size_t j = 1; j < M + 1; ++j) {
            int r = padded_img_c1(i, j);
            std::bitset<8> binary_neigh1 =
                neigh_matrix_lex(padded_img_c1, i, j, r);

            std::vector<int> thresholds2(10, T2 + 1); // can be an array
            int s = padded_img_c2(i, j);
            for (size_t p = 0; p < 3; ++p) {
                for (size_t q = 0; q < 3; ++q) {
                    int val = padded_img_c2(i - 1 + p, j - 1 + q);
                    if (val >= s)
                        thresholds2[q + p * 3] = val;
                }
            }

            sort(thresholds2.begin(), thresholds2.end());
            auto last = unique(thresholds2.begin(), thresholds2.end());
            thresholds2.erase(last, thresholds2.end());

            int last_change = 0;
            for (size_t z = 0; z < thresholds2.size() - 1; ++z) {
                size_t start = static_cast<size_t>(thresholds2[z]);
                size_t end = static_cast<size_t>(thresholds2[z + 1]);

                std::bitset<8> binary_neigh2 =
                    neigh_matrix(padded_img_c2, i, j, thresholds2[z]);
                std::bitset<8> binary_neigh = binary_neigh1 & binary_neigh2;

                size_t neigh_num = binary_neigh.to_ulong();
                int change = ec(neigh_num);

                ecp(static_cast<int>(r), start) += change - last_change;
                last_change = change;
            }
        }
    }

    for (int t = 0; t <= T1; ++t) {
        for (int s = 1; s <= T2; ++s) {
            ecp(t, s) += ecp(t, s - 1);
        }
    }

    for (size_t s = 1; s < T1 + 1; ++s) {
        for (size_t t = 0; t < T2 + 1; ++t) {
            ecp(s, t) += ecp(s - 1, t);
        }
    }

    /*
        auto capsule = py::capsule(ecp, [](void *f) {
            delete reinterpret_cast<Matrix<int> *>(f);
        });
    */

    return py::array_t<int>({ecp.dim(0), ecp.dim(1)}, // Shape
                            ecp.data_ptr());
}

int euler_change_neigh_matrix(Matrix<uint8_t, 2> neigh) {
    uint8_t edges = neigh(0, 1) + neigh(1, 2) + neigh(2, 1) + neigh(1, 0);
    uint8_t vA = neigh(1, 0) || neigh(0, 0) || neigh(0, 1);
    uint8_t vB = neigh(0, 1) || neigh(0, 2) || neigh(1, 2);
    uint8_t vC = neigh(1, 2) || neigh(2, 2) || neigh(2, 1);
    uint8_t vD = neigh(2, 1) || neigh(2, 0) || neigh(1, 0);
    uint8_t vertices = vA + vB + vC + vD;
    return 1 - vertices + edges;
}

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

Matrix<uint8_t, 2> py_binary_threshold_image_2d3c(py::array_t<int> image_c1,
                                                  py::array_t<int> image_c2,
                                                  py::array_t<int> image_c3,
                                                  int r, int s, int t) {
    auto img_c1 = image_c1.unchecked<2>();
    auto img_c2 = image_c2.unchecked<2>();
    auto img_c3 = image_c3.unchecked<2>();

    size_t num_rows = img_c1.shape(0);
    size_t num_cols = img_c2.shape(1);

    Matrix<uint8_t, 2> thresholded_img(num_rows, num_cols, 0);

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < num_cols; j++) {
            thresholded_img(i, j) =
                img_c1(i, j) <= r && img_c2(i, j) <= s && img_c3(i, j) <= t;
        }
    }

    return thresholded_img;
}

py::array_t<int> py_ecp_2d3c_naive(py::array_t<int> image_c1,
                                   py::array_t<int> image_c2,
                                   py::array_t<int> image_c3, int T1, int T2,
                                   int T3) {
    Matrix<int, 3> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 1),
                       static_cast<size_t>(T3 + 1), 0);

    for (int r = 0; r <= T1; r++) {
        Matrix<uint8_t, 2> thresholded_img_c1 =
            py_binary_threshold_image_2d(image_c1, r);

        for (int s = 0; s <= T2; s++) {
            Matrix<uint8_t, 2> thresholded_img_c2 =
                py_binary_threshold_image_2d(image_c2, s);

            for (int t = 0; t <= T3; t++) {
                Matrix<uint8_t, 2> thresholded_img_c3 =
                    py_binary_threshold_image_2d(image_c3, t);
                Matrix<uint8_t, 2> K_rst = elementwise_AND_2d(
                    thresholded_img_c1, thresholded_img_c2, thresholded_img_c3);
                ecp(r, s, t) = ec_binary_image_2d_naive(K_rst);
            }
        }
    }

    return py::array_t<int>({ecp.dim(0), ecp.dim(1), ecp.dim(2)},
                            ecp.data_ptr());
}

py::array_t<int> py_ecp_2d3c_naive2(py::array_t<int> image_c1,
                                    py::array_t<int> image_c2,
                                    py::array_t<int> image_c3, int T1, int T2,
                                    int T3) {
    Matrix<int, 3> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 1),
                       static_cast<size_t>(T3 + 1), 0);

    for (int r = 0; r <= T1; r++) {
        for (int s = 0; s <= T2; s++) {
            for (int t = 0; t <= T3; t++) {
                Matrix<uint8_t, 2> K_rst = py_binary_threshold_image_2d3c(
                    image_c1, image_c2, image_c3, r, s, t);
                ecp(r, s, t) = ec_binary_image_2d_naive(K_rst);
            }
        }
    }

    return py::array_t<int>({ecp.dim(0), ecp.dim(1), ecp.dim(2)},
                            ecp.data_ptr());
}

py::array_t<int> py_ecp_2d3c(py::array_t<int> image_c1,
                             py::array_t<int> image_c2,
                             py::array_t<int> image_c3,
                             py::array_t<int> euler_changes, int T1, int T2,
                             int T3) {
    auto ec = euler_changes.unchecked<1>();

    size_t N = image_c1.shape(0);
    size_t M = image_c1.shape(1);

    Matrix<int, 3> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 1),
                       static_cast<size_t>(T3 + 1), 0);

    Matrix<int, 2> padded_img_c1 = pad_img_2d(image_c1, T1 + 1);
    Matrix<int, 2> padded_img_c2 = pad_img_2d(image_c2, T2 + 1);
    Matrix<int, 2> padded_img_c3 = pad_img_2d(image_c3, T3 + 1);

    for (size_t i = 1; i < N + 1; ++i) {
        for (size_t j = 1; j < M + 1; ++j) {
            int r = padded_img_c1(i, j);
            std::bitset<8> binary_neigh1 =
                neigh_matrix_lex(padded_img_c1, i, j, r);

            std::vector<int> thresholds2(10, T2 + 1); // can be an array
            int s = padded_img_c2(i, j);
            for (size_t p = 0; p < 3; ++p) {
                for (size_t q = 0; q < 3; ++q) {
                    int val = padded_img_c2(i - 1 + p, j - 1 + q);
                    if (val >= s)
                        thresholds2[q + p * 3] = val;
                }
            }
            sort(thresholds2.begin(), thresholds2.end());
            auto last = unique(thresholds2.begin(), thresholds2.end());
            thresholds2.erase(last, thresholds2.end());

            std::vector<int> thresholds3(10, T3 + 1);
            int t = padded_img_c3(i, j);
            for (size_t p = 0; p < 3; ++p) {
                for (size_t q = 0; q < 3; ++q) {
                    int val = padded_img_c3(i - 1 + p, j - 1 + q);
                    if (val >= t)
                        thresholds3[q + p * 3] = val;
                }
            }
            sort(thresholds3.begin(), thresholds3.end());
            last = unique(thresholds3.begin(), thresholds3.end());
            thresholds3.erase(last, thresholds3.end());

            for (size_t z = 0; z < thresholds2.size() - 1; ++z) {
                size_t start2 = static_cast<size_t>(thresholds2[z]);
                size_t end2 = static_cast<size_t>(thresholds2[z + 1]);

                std::bitset<8> binary_neigh2 =
                    neigh_matrix(padded_img_c2, i, j, thresholds2[z]);

                for (size_t y = 0; y < thresholds3.size() - 1; ++y) {
                    size_t start3 = static_cast<size_t>(thresholds3[y]);
                    size_t end3 = static_cast<size_t>(thresholds3[y + 1]);

                    std::bitset<8> binary_neigh3 =
                        neigh_matrix(padded_img_c3, i, j, thresholds3[y]);
                    std::bitset<8> binary_neigh =
                        binary_neigh1 & binary_neigh2 & binary_neigh3;

                    size_t neigh_num = binary_neigh.to_ulong();
                    int change = ec(neigh_num);

                    for (size_t ind2 = start2; ind2 < end2; ++ind2) {
                        for (size_t ind3 = start3; ind3 < end3; ++ind3) {
                            ecp(static_cast<int>(r), ind2, ind3) += change;
                        }
                    }
                }
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

    return py::array_t<int>({ecp.dim(0), ecp.dim(1), ecp.dim(2)}, // Shape
                            ecp.data_ptr());
}

py::array_t<int> py_ecp_2d3c_optimized(py::array_t<int> image_c1,
                                       py::array_t<int> image_c2,
                                       py::array_t<int> image_c3,
                                       py::array_t<int> euler_changes, int T1,
                                       int T2, int T3) {
    auto ec = euler_changes.unchecked<1>();

    size_t N = image_c1.shape(0);
    size_t M = image_c1.shape(1);

    Matrix<int, 3> ecp(static_cast<size_t>(T1 + 1), static_cast<size_t>(T2 + 2),
                       static_cast<size_t>(T3 + 2), 0);

    Matrix<int, 2> padded_img_c1 = pad_img_2d(image_c1, T1 + 1);
    Matrix<int, 2> padded_img_c2 = pad_img_2d(image_c2, T2 + 1);
    Matrix<int, 2> padded_img_c3 = pad_img_2d(image_c3, T3 + 1);

    for (size_t i = 1; i < N + 1; ++i) {
        for (size_t j = 1; j < M + 1; ++j) {
            int r = padded_img_c1(i, j);
            std::bitset<8> binary_neigh1 =
                neigh_matrix_lex(padded_img_c1, i, j, r);

            std::vector<int> thresholds2(10, T2 + 1); // can be an array
            int s = padded_img_c2(i, j);
            for (size_t p = 0; p < 3; ++p) {
                for (size_t q = 0; q < 3; ++q) {
                    int val = padded_img_c2(i - 1 + p, j - 1 + q);
                    if (val >= s)
                        thresholds2[q + p * 3] = val;
                }
            }
            sort(thresholds2.begin(), thresholds2.end());
            auto last = unique(thresholds2.begin(), thresholds2.end());
            thresholds2.erase(last, thresholds2.end());

            std::vector<int> thresholds3(10, T3 + 1);
            int t = padded_img_c3(i, j);
            for (size_t p = 0; p < 3; ++p) {
                for (size_t q = 0; q < 3; ++q) {
                    int val = padded_img_c3(i - 1 + p, j - 1 + q);
                    if (val >= t)
                        thresholds3[q + p * 3] = val;
                }
            }
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
                    size_t end3 = static_cast<size_t>(thresholds3[y + 1]);

                    std::bitset<8> binary_neigh3 =
                        neigh_matrix(padded_img_c3, i, j, thresholds3[y]);
                    std::bitset<8> binary_neigh =
                        binary_neigh1 & binary_neigh2 & binary_neigh3;

                    size_t neigh_num = binary_neigh.to_ulong();
                    int change = ec(neigh_num);

                    ecp(static_cast<int>(r), start2, start3) +=
                        change - last_change;
                    ecp(static_cast<int>(r), end2, start3) -=
                        change - last_change;
                    last_change = change;
                }

                ecp(static_cast<int>(r), start2, T3 + 1) -= last_change;
                ecp(static_cast<int>(r), end2, T3 + 1) += last_change;
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

// Heiss & Wagner
py::array_t<int> py_ecp_2d2c_hw(py::array_t<int> image_c1,
                                py::array_t<int> image_c2, int T1, int T2) {
    Matrix<int, 2> pad_img_c1 = pad_img_2d(image_c1, T1 + 1);
    Matrix<int, 2> pad_img_c2 = pad_img_2d(image_c2, T2 + 1);

    size_t num_rows = image_c2.shape(0);
    size_t num_cols = image_c2.shape(1);

    Matrix<int, 2> ecp(T1 + 1, T2 + 1, 0);

    for (size_t s = 0; s <= T2; ++s) {
        for (size_t i = 1; i < num_rows + 1; ++i) {
            for (size_t j = 1; j < num_cols + 1; ++j) {
                if (pad_img_c2(i, j) <= s) {
                    int val = pad_img_c1(i, j);

                    int n1 = pad_img_c2(i - 1, j - 1) <= s
                                 ? pad_img_c1(i - 1, j - 1)
                                 : T1 + 1;
                    int n2 = pad_img_c2(i - 1, j) <= s ? pad_img_c1(i - 1, j)
                                                       : T1 + 1;
                    int n3 = pad_img_c2(i - 1, j + 1) <= s
                                 ? pad_img_c1(i - 1, j + 1)
                                 : T1 + 1;
                    int n4 = pad_img_c2(i, j - 1) <= s ? pad_img_c1(i, j - 1)
                                                       : T1 + 1;
                    int n6 = pad_img_c2(i, j + 1) <= s ? pad_img_c1(i, j + 1)
                                                       : T1 + 1;
                    int n7 = pad_img_c2(i + 1, j - 1) <= s
                                 ? pad_img_c1(i + 1, j - 1)
                                 : T1 + 1;
                    int n8 = pad_img_c2(i + 1, j) <= s ? pad_img_c1(i + 1, j)
                                                       : T1 + 1;
                    int n9 = pad_img_c2(i + 1, j + 1) <= s
                                 ? pad_img_c1(i + 1, j + 1)
                                 : T1 + 1;

                    int change = 0;

                    // square
                    change += 1;

                    // edges
                    change -= n2 >= val; // top edge
                    change -= n6 > val;  // right edge
                    change -= n8 > val;  // bottom edge
                    change -= n4 >= val; // left edge

                    // vertices
                    change +=
                        n1 >= val && n2 >= val && n4 >= val; // top-left vertex
                    change +=
                        n2 >= val && n3 >= val && n6 > val; // top-right vertex
                    change +=
                        n4 >= val && n7 > val && n8 > val; // bottom-left vertex
                    change +=
                        n6 > val && n8 > val && n9 > val; // bottom-right vertex

                    ecp(val, s) += change;
                }
            }
        }

        // cummulative sum
        for (size_t r = 1; r <= T1; ++r) {
            ecp(r, s) += ecp(r - 1, s);
        }
    }

    return py::array_t<int>({ecp.dim(0), ecp.dim(1)}, ecp.data_ptr());
}

py::array_t<int> py_ecp_2d2c_hw_optimized(py::array_t<int> image_c1,
                                          py::array_t<int> image_c2, int T1,
                                          int T2) {
    Matrix<int, 2> pad_img_c1 = pad_img_2d(image_c1, T1 + 1);
    Matrix<int, 2> pad_img_c2 = pad_img_2d(image_c2, T2 + 1);

    size_t num_rows = image_c2.shape(0);
    size_t num_cols = image_c2.shape(1);

    Matrix<int, 2> ecp(T1 + 1, T2 + 1, 0);

    for (size_t i = 1; i < num_rows + 1; ++i) {
        for (size_t j = 1; j < num_cols + 1; ++j) {
            std::vector<int> thresholds2(10, T2 + 1); // can be an array
            int s = pad_img_c2(i, j);
            for (size_t p = 0; p < 3; ++p) {
                for (size_t q = 0; q < 3; ++q) {
                    int val = pad_img_c2(i - 1 + p, j - 1 + q);
                    if (val >= s)
                        thresholds2[q + p * 3] = val;
                }
            }
            sort(thresholds2.begin(), thresholds2.end());
            auto last = unique(thresholds2.begin(), thresholds2.end());
            thresholds2.erase(last, thresholds2.end());

            for (size_t z = 0; z < thresholds2.size() - 1; ++z) {
                size_t start = static_cast<size_t>(thresholds2[z]);
                size_t end = static_cast<size_t>(thresholds2[z + 1]);

                int s = thresholds2[z];

                int val = pad_img_c1(i, j);

                int n1 = pad_img_c2(i - 1, j - 1) <= s
                             ? pad_img_c1(i - 1, j - 1)
                             : T1 + 1;
                int n2 =
                    pad_img_c2(i - 1, j) <= s ? pad_img_c1(i - 1, j) : T1 + 1;
                int n3 = pad_img_c2(i - 1, j + 1) <= s
                             ? pad_img_c1(i - 1, j + 1)
                             : T1 + 1;
                int n4 =
                    pad_img_c2(i, j - 1) <= s ? pad_img_c1(i, j - 1) : T1 + 1;
                int n6 =
                    pad_img_c2(i, j + 1) <= s ? pad_img_c1(i, j + 1) : T1 + 1;
                int n7 = pad_img_c2(i + 1, j - 1) <= s
                             ? pad_img_c1(i + 1, j - 1)
                             : T1 + 1;
                int n8 =
                    pad_img_c2(i + 1, j) <= s ? pad_img_c1(i + 1, j) : T1 + 1;
                int n9 = pad_img_c2(i + 1, j + 1) <= s
                             ? pad_img_c1(i + 1, j + 1)
                             : T1 + 1;

                int change = 0;

                // square
                change += 1;

                // edges
                change -= n2 >= val; // top edge
                change -= n6 > val;  // right edge
                change -= n8 > val;  // bottom edge
                change -= n4 >= val; // left edge

                // vertices
                change +=
                    n1 >= val && n2 >= val && n4 >= val; // top-left vertex
                change +=
                    n2 >= val && n3 >= val && n6 > val; // top-right vertex
                change +=
                    n4 >= val && n7 > val && n8 > val; // bottom-left vertex
                change +=
                    n6 > val && n8 > val && n9 > val; // bottom-right vertex

                for (size_t ind = start; ind < end; ++ind) {
                    ecp(static_cast<int>(val), ind) += change;
                }
            }
        }
    }

    // cummulative sum
    for (size_t s = 0; s <= T2; ++s) {
        for (size_t r = 1; r <= T1; ++r) {
            ecp(r, s) += ecp(r - 1, s);
        }
    }

    return py::array_t<int>({ecp.dim(0), ecp.dim(1)}, ecp.data_ptr());
}

py::array_t<int> py_ecp_2d2c_hw_optimized2(py::array_t<int> image_c1,
                                           py::array_t<int> image_c2, int T1,
                                           int T2) {
    Matrix<int, 2> pad_img_c1 = pad_img_2d(image_c1, T1 + 1);
    Matrix<int, 2> pad_img_c2 = pad_img_2d(image_c2, T2 + 1);

    size_t num_rows = image_c2.shape(0);
    size_t num_cols = image_c2.shape(1);

    // 1. Allocate with an extra column (T2 + 2) to handle the 'end' marker
    // safely
    // 2. We use a raw buffer or std::vector to ensure we can hand off ownership
    // to Python
    std::vector<int> ecp_buffer((T1 + 1) * (T2 + 2), 0);
    auto get_ecp = [&](int v, int s) -> int & {
        return ecp_buffer[v * (T2 + 2) + s];
    };

    for (size_t i = 1; i <= num_rows; ++i) {
        for (size_t j = 1; j <= num_cols; ++j) {
            int s_center = pad_img_c2(i, j);
            if (s_center > T2)
                continue;

            int val_c1 = pad_img_c1(i, j);
            if (val_c1 > T1)
                continue;

            // Collect unique thresholds in the 3x3 neighborhood
            int thresholds[10];
            int count = 0;
            for (int p = -1; p <= 1; ++p) {
                for (int q = -1; q <= 1; ++q) {
                    int s_val = pad_img_c2(i + p, j + q);
                    // Only care about transitions that happen AFTER or AT the
                    // center pixel's start
                    if (s_val >= s_center && s_val <= T2) {
                        thresholds[count++] = s_val;
                    }
                }
            }
            thresholds[count++] = T2 + 1; // The "turn off" event

            std::sort(thresholds, thresholds + count);
            int *end_ptr = std::unique(thresholds, thresholds + count);
            int unique_count = static_cast<int>(end_ptr - thresholds);

            int last_change = 0;
            for (int z = 0; z < unique_count - 1; ++z) {
                int s = thresholds[z];

                auto get_n = [&](int di, int dj) {
                    // Logic: if neighbor is active at 's', get its c1 value,
                    // else T1+1
                    return (pad_img_c2(i + di, j + dj) <= s)
                               ? pad_img_c1(i + di, j + dj)
                               : T1 + 1;
                };

                int n1 = get_n(-1, -1), n2 = get_n(-1, 0), n3 = get_n(-1, 1);
                int n4 = get_n(0, -1), n6 = get_n(0, 1);
                int n7 = get_n(1, -1), n8 = get_n(1, 0), n9 = get_n(1, 1);

                int change = 1;
                change -= (n2 >= val_c1) + (n6 > val_c1) + (n8 > val_c1) +
                          (n4 >= val_c1);
                change += (n1 >= val_c1 && n2 >= val_c1 && n4 >= val_c1);
                change += (n2 >= val_c1 && n3 >= val_c1 && n6 > val_c1);
                change += (n4 >= val_c1 && n7 > val_c1 && n8 > val_c1);
                change += (n6 > val_c1 && n8 > val_c1 && n9 > val_c1);

                // Apply Difference Array logic
                int delta = change - last_change;
                get_ecp(val_c1, s) += delta;
                last_change = change;
            }
            // Mark the end of the last interval
            get_ecp(val_c1, thresholds[unique_count - 1]) -= last_change;
        }
    }

    // Pass 1: Integrate over S (smears the difference markers)
    for (int v = 0; v <= T1; ++v) {
        for (int s = 1; s <= T2; ++s) {
            get_ecp(v, s) += get_ecp(v, s - 1);
        }
    }

    // Pass 2: Integrate over T1 (Your original cumulative sum)
    for (int s = 0; s <= T2; ++s) {
        for (int v = 1; v <= T1; ++v) {
            get_ecp(v, s) += get_ecp(v - 1, s);
        }
    }

    // Create the result array.
    // IMPORTANT: We copy the data into a new py::array_t to avoid ownership
    // issues.
    py::array_t<int> result({(size_t)T1 + 1, (size_t)T2 + 1});
    auto r = result.mutable_unchecked<2>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            r(v, s) = get_ecp(v, s);
        }
    }

    return result;
}

py::array_t<int> py_ecp_2d3c_hw_optimized(py::array_t<int> image_c1,
                                          py::array_t<int> image_c2,
                                          py::array_t<int> image_c3, int T1,
                                          int T2, int T3) {
    // 1. Padding
    auto pad_c1 = pad_img_2d(image_c1, T1 + 1);
    auto pad_c2 = pad_img_2d(image_c2, T2 + 1);
    auto pad_c3 = pad_img_2d(image_c3, T3 + 1);

    size_t rows = image_c2.shape(0);
    size_t cols = image_c2.shape(1);

    // 2. Allocate 3D Difference Array
    // Size is (T1+1) * (T2+2) * (T3+2) to handle "exit" markers safely
    size_t dim1 = T1 + 1;
    size_t dim2 = T2 + 2;
    size_t dim3 = T3 + 2;
    std::vector<int> ecp_vol(dim1 * dim2 * dim3, 0);

    auto add_diff = [&](int v, int s, int t, int delta) {
        ecp_vol[(v * dim2 * dim3) + (s * dim3) + t] += delta;
    };

    // 3. Main Image Pass
    for (size_t i = 1; i <= rows; ++i) {
        for (size_t j = 1; j <= cols; ++j) {
            int v_val = pad_c1(i, j);
            int s_start = pad_c2(i, j);
            int t_start = pad_c3(i, j);

            if (v_val > T1 || s_start > T2 || t_start > T3)
                continue;

            // This algorithm by Gemini probably uses BACKWARD differences while
            // I am developing the theory for FORWARD differences.
            std::vector<int> s_jumps = {
                s_start,
                T2 + 1}; // I think now it can just be T2 instead of T2+1
            std::vector<int> t_jumps = {t_start,
                                        T3 + 1}; // same for T3 instead of T3+1

            for (int p = -1; p <= 1; ++p) {
                for (int q = -1; q <= 1; ++q) {
                    int s_n = pad_c2(i + p, j + q);
                    int t_n = pad_c3(i + p, j + q);
                    if (s_n > s_start && s_n <= T2)
                        s_jumps.push_back(s_n);
                    if (t_n > t_start && t_n <= T3)
                        t_jumps.push_back(t_n);
                }
            }

            std::sort(s_jumps.begin(), s_jumps.end());
            s_jumps.erase(std::unique(s_jumps.begin(), s_jumps.end()),
                          s_jumps.end());

            std::sort(t_jumps.begin(), t_jumps.end());
            t_jumps.erase(std::unique(t_jumps.begin(), t_jumps.end()),
                          t_jumps.end());

            // 4. Local Grid Evaluation
            // We use 2D difference logic on the (S, T) plane for this V level
            int last_row_change = 0;
            for (size_t sz = 0; sz < s_jumps.size() - 1; ++sz) {
                int s_curr = s_jumps[sz];
                int s_next = s_jumps[sz + 1];
                int last_patch_change = 0;

                for (size_t tz = 0; tz < t_jumps.size() - 1; ++tz) {
                    int t_curr = t_jumps[tz];
                    int t_next = t_jumps[tz + 1];

                    auto is_active = [&](int di, int dj) {
                        return pad_c2(i + di, j + dj) <= s_curr &&
                               pad_c3(i + di, j + dj) <= t_curr;
                    };

                    auto get_n = [&](int di, int dj) {
                        return is_active(di, dj) ? pad_c1(i + di, j + dj)
                                                 : T1 + 1;
                    };

                    // Standard Euler calculation
                    int n1 = get_n(-1, -1), n2 = get_n(-1, 0),
                        n3 = get_n(-1, 1);
                    int n4 = get_n(0, -1), n6 = get_n(0, 1);
                    int n7 = get_n(1, -1), n8 = get_n(1, 0), n9 = get_n(1, 1);

                    int change = 1;
                    // edges
                    change -= (n2 >= v_val) + (n6 > v_val) + (n8 > v_val) +
                              (n4 >= v_val);
                    // vertices
                    change += (n1 >= v_val && n2 >= v_val && n4 >= v_val);
                    change += (n2 >= v_val && n3 >= v_val && n6 > v_val);
                    change += (n4 >= v_val && n7 > v_val && n8 > v_val);
                    change += (n6 > v_val && n8 > v_val && n9 > v_val);

                    // Apply 2D Difference markers to the (S, T) slice at this V
                    int delta = change - last_patch_change;
                    if (delta != 0) {
                        add_diff(v_val, s_curr, t_curr, delta);
                        add_diff(v_val, s_next, t_curr, -delta);
                    }
                    last_patch_change = change;
                    /*
                    add_diff(v_val, s_curr, t_curr, change);
                    add_diff(v_val, s_next, t_curr, -change);
                    add_diff(v_val, s_curr, t_next, -change);
                    add_diff(v_val, s_next, t_next, change);
                    */
                }
                // Close the final T intervals for this S row
                if (last_patch_change != 0) {
                    add_diff(v_val, s_curr, T3 + 1, -last_patch_change);
                    add_diff(v_val, s_next, T3 + 1, last_patch_change);
                }
            }
        }
    }

    // 5. Triple Integration
    // Pass 1: Over T
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 1; t <= T3; ++t) {
                ecp_vol[v * dim2 * dim3 + s * dim3 + t] +=
                    ecp_vol[v * dim2 * dim3 + s * dim3 + (t - 1)];
            }
        }
    }
    // Pass 2: Over S
    for (int v = 0; v <= T1; ++v) {
        for (int s = 1; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp_vol[v * dim2 * dim3 + s * dim3 + t] +=
                    ecp_vol[v * dim2 * dim3 + (s - 1) * dim3 + t];
            }
        }
    }
    // Pass 3: Over V (Cumulative over C1)
    for (int v = 1; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp_vol[v * dim2 * dim3 + s * dim3 + t] +=
                    ecp_vol[(v - 1) * dim2 * dim3 + s * dim3 + t];
            }
        }
    }

    /*
    // Pass 1: Over T (The fastest pass - contiguous memory)
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            // Point to the start of the 't' line for this specific v and s
            int* t_line = &ecp_vol[v * dim2 * dim3 + s * dim3];
            for (int t = 1; t <= T3; ++t) {
                t_line[t] += t_line[t - 1];
            }
        }
    }

    // Pass 2: Over S (Strided access - one 'dim3' jump per step)
    for (int v = 0; v <= T1; ++v) {
        int* v_plane = &ecp_vol[v * dim2 * dim3];
        for (int t = 0; t <= T3; ++t) {
            // We walk 's' for a fixed 't'.
            // Each increment of 's' is a jump of 'dim3' in the flat array.
            for (int s = 1; s <= T2; ++s) {
                v_plane[s * dim3 + t] += v_plane[(s - 1) * dim3 + t];
            }
        }
    }

    // Pass 3: Over V (Large strides - one 'dim2 * dim3' jump per step)
    int plane_size = dim2 * dim3;
    for (int s = 0; s <= T2; ++s) {
        int offset_st = s * dim3;
        for (int t = 0; t <= T3; ++t) {
            int final_offset = offset_st + t;
            for (int v = 1; v <= T1; ++v) {
                // Accessing the same (s,t) position across different 'v' planes
                ecp_vol[v * plane_size + final_offset] += ecp_vol[(v - 1) *
    plane_size + final_offset];
            }
        }
    }
    */

    // 6. Return as 3D NumPy Array
    py::array_t<int> result({(size_t)T1 + 1, (size_t)T2 + 1, (size_t)T3 + 1});
    auto r = result.mutable_unchecked<3>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                r(v, s, t) = ecp_vol[v * dim2 * dim3 + s * dim3 + t];
            }
        }
    }

    return result;
}

py::array_t<int> py_ecp_2d3c_hw_optimized_test(py::array_t<int> image_c1,
                                               py::array_t<int> image_c2,
                                               py::array_t<int> image_c3,
                                               int T1, int T2, int T3) {
    // 1. Padding
    auto pad_c1 = pad_img_2d(image_c1, T1 + 1);
    auto pad_c2 = pad_img_2d(image_c2, T2 + 1);
    auto pad_c3 = pad_img_2d(image_c3, T3 + 1);

    size_t rows = image_c2.shape(0);
    size_t cols = image_c2.shape(1);

    // 2. Allocate 3D Difference Array
    // Size is (T1+1) * (T2+2) * (T3+2) to handle "exit" markers safely
    size_t dim1 = T1 + 1;
    size_t dim2 = T2 + 1;
    size_t dim3 = T3 + 1;
    std::vector<int> ecp_vol(dim1 * dim2 * dim3, 0);

    auto add_diff = [&](int v, int s, int t, int delta) {
        ecp_vol[(v * dim2 * dim3) + (s * dim3) + t] += delta;
    };

    // 3. Main Image Pass
    for (size_t i = 1; i <= rows; ++i) {
        for (size_t j = 1; j <= cols; ++j) {
            int v_val = pad_c1(i, j);
            int s_start = pad_c2(i, j);
            int t_start = pad_c3(i, j);

            if (v_val > T1 || s_start > T2 || t_start > T3)
                continue;

            // This algorithm by Gemini probably uses BACKWARD differences while
            // I am developing the theory for FORWARD differences.
            std::vector<int> s_jumps = {
                s_start,
                T2 + 1}; // I think now it can just be T2 instead of T2+1
            std::vector<int> t_jumps = {t_start,
                                        T3 + 1}; // same for T3 instead of T3+1

            for (int p = -1; p <= 1; ++p) {
                for (int q = -1; q <= 1; ++q) {
                    int s_n = pad_c2(i + p, j + q);
                    int t_n = pad_c3(i + p, j + q);
                    if (s_n > s_start && s_n <= T2)
                        s_jumps.push_back(s_n);
                    if (t_n > t_start && t_n <= T3)
                        t_jumps.push_back(t_n);
                }
            }

            std::sort(s_jumps.begin(), s_jumps.end());
            s_jumps.erase(std::unique(s_jumps.begin(), s_jumps.end()),
                          s_jumps.end());

            std::sort(t_jumps.begin(), t_jumps.end());
            t_jumps.erase(std::unique(t_jumps.begin(), t_jumps.end()),
                          t_jumps.end());

            // 4. Local Grid Evaluation
            // We use 2D difference logic on the (S, T) plane for this V level
            int last_row_change = 0;
            for (size_t sz = 0; sz < s_jumps.size() - 1; ++sz) {
                int s_curr = s_jumps[sz];
                int s_next = s_jumps[sz + 1];
                int last_patch_change = 0;

                for (size_t tz = 0; tz < t_jumps.size() - 1; ++tz) {
                    int t_curr = t_jumps[tz];
                    int t_next = t_jumps[tz + 1];

                    auto is_active = [&](int di, int dj) {
                        return pad_c2(i + di, j + dj) <= s_curr &&
                               pad_c3(i + di, j + dj) <= t_curr;
                    };

                    auto get_n = [&](int di, int dj) {
                        return is_active(di, dj) ? pad_c1(i + di, j + dj)
                                                 : T1 + 1;
                    };

                    // Standard Euler calculation
                    int n1 = get_n(-1, -1), n2 = get_n(-1, 0),
                        n3 = get_n(-1, 1);
                    int n4 = get_n(0, -1), n6 = get_n(0, 1);
                    int n7 = get_n(1, -1), n8 = get_n(1, 0), n9 = get_n(1, 1);

                    int change = 1;
                    change -= (n2 >= v_val) + (n6 > v_val) + (n8 > v_val) +
                              (n4 >= v_val);
                    change += (n1 >= v_val && n2 >= v_val && n4 >= v_val);
                    change += (n2 >= v_val && n3 >= v_val && n6 > v_val);
                    change += (n4 >= v_val && n7 > v_val && n8 > v_val);
                    change += (n6 > v_val && n8 > v_val && n9 > v_val);

                    add_diff(v_val, s_curr, t_curr, change);
                    bool not_border_s = s_next != T2 + 1;
                    bool not_border_t = t_next != T3 + 1;
                    if (not_border_s)
                        add_diff(v_val, s_next, t_curr, -change);
                    if (not_border_t)
                        add_diff(v_val, s_curr, t_next, -change);
                    if (not_border_s && not_border_t)
                        add_diff(v_val, s_next, t_next, change);
                }
            }
        }
    }

    // 5. Triple Integration
    // Pass 1: Over T
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 1; t <= T3; ++t) {
                ecp_vol[v * dim2 * dim3 + s * dim3 + t] +=
                    ecp_vol[v * dim2 * dim3 + s * dim3 + (t - 1)];
            }
        }
    }
    // Pass 2: Over S
    for (int v = 0; v <= T1; ++v) {
        for (int s = 1; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp_vol[v * dim2 * dim3 + s * dim3 + t] +=
                    ecp_vol[v * dim2 * dim3 + (s - 1) * dim3 + t];
            }
        }
    }
    // Pass 3: Over V (Cumulative over C1)
    for (int v = 1; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                ecp_vol[v * dim2 * dim3 + s * dim3 + t] +=
                    ecp_vol[(v - 1) * dim2 * dim3 + s * dim3 + t];
            }
        }
    }

    // 6. Return as 3D NumPy Array
    py::array_t<int> result({(size_t)T1 + 1, (size_t)T2 + 1, (size_t)T3 + 1});
    auto r = result.mutable_unchecked<3>();
    for (int v = 0; v <= T1; ++v) {
        for (int s = 0; s <= T2; ++s) {
            for (int t = 0; t <= T3; ++t) {
                r(v, s, t) = ecp_vol[v * dim2 * dim3 + s * dim3 + t];
            }
        }
    }

    return result;
}

bool is_pointwise_seq_3d(std::tuple<int, int, int> &a,
                         std::tuple<int, int, int> &b) {
    return std::get<0>(a) <= std::get<0>(b) &&
           std::get<1>(a) <= std::get<1>(b) && std::get<2>(a) <= std::get<2>(b);
}

bool is_pointwise_sm_3d(std::tuple<int, int, int> &a,
                        std::tuple<int, int, int> &b) {
    return std::get<0>(a) <= std::get<0>(b) &&
           std::get<1>(a) <= std::get<1>(b) &&
           std::get<2>(a) <= std::get<2>(b) && a != b;
}

bool are_comparable(std::tuple<int, int, int> &a,
                    std::tuple<int, int, int> &b) {
    return is_pointwise_seq_3d(a, b) || is_pointwise_seq_3d(b, a);
}

std::tuple<int, int, int> pointwise_max(std::tuple<int, int, int> &a,
                                        std::tuple<int, int, int> &b) {
    return std::tuple<int, int, int>{};
}

size_t partition_comparable(std::vector<std::tuple<int, int, int>> &L) {
    size_t N = L.size();

    std::vector<bool> keep(N, true);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            if (is_pointwise_seq_3d(L[i], L[j])) {
                keep[j] = false;
            } else if (is_pointwise_seq_3d(L[j], L[i])) {
                keep[i] = false;
            }
        }
    }

    std::cout << "[";
    for (size_t i = 0; i < N - 1; ++i) {
        std::cout << keep[i] << ", ";
    }
    std::cout << keep[N - 1] << "]" << std::endl;

    size_t insert_pos = 0;
    for (size_t i = 0; i < N; ++i) {
        if (keep[i]) {
            std::swap(L[i], L[insert_pos]);
            insert_pos++;
        }
    }

    return insert_pos;
}

int max(int a, int b) { return (a <= b) ? b : a; }

std::vector<std::tuple<int, int, int, int>> compute_contributions_multicritical(
    size_t dim, std::vector<std::tuple<int, int, int>> non_comparable) {
    std::vector<std::tuple<int, int, int, int>> contributions;

    int base_contribution = (dim % 2 == 0) ? 1 : -1;

    for (auto t : non_comparable) {
        contributions.push_back({std::get<0>(t), std::get<1>(t), std::get<2>(t),
                                 base_contribution});
    }

    std::vector<std::tuple<int, int, int>> P;
    for (size_t i = 0; i < non_comparable.size(); ++i) {
        for (size_t j = i; j < non_comparable.size(); ++j) {
            P.push_back(max(non_comparable[i], non_comparable[j]));
        }
    }

    std::queue<std::tuple<int, int, int>> L;
    for (size_t i = 0; i < L.size(); ++i) {
        for (size_t j = i + 1; j < L.size(); ++j) {
            // P.push_back(max(L[i],L[j]));
        }
    }

    while (!L.empty()) {
        std::tuple<int, int, int> p = L.front();
        L.pop();

        std::vector<std::tuple<int, int, int>> P_prime;
        for (auto t : P) {
            if (is_pointwise_sm_3d(t, p)) {
                P_prime.push_back(t);
            }
        }
    }
}

py::array_t<int> py_ecp_2d3c_dg(py::array_t<int> image_c1,
                                py::array_t<int> image_c2,
                                py::array_t<int> image_c3, int T1, int T2,
                                int T3) {
    auto img_c1 = image_c1.unchecked<2>();
    auto img_c2 = image_c2.unchecked<2>();
    auto img_c3 = image_c3.unchecked<2>();

    size_t num_rows = img_c1.shape(0);
    size_t num_cols = img_c2.shape(1);

    std::vector<std::tuple<int, int, int, int>> contributions;

    for (size_t i = 1; i < num_rows - 1; ++i) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            contributions.push_back({img_c1(i, j), img_c2(i, j), img_c3(i, j),
                                     1}); // the square itself

            // top-right vertex

            // top edge

            // right edge
        }
    }
}

// Computes the ECP for the TOP-CELL FILTRATION not the filtration induced by
// top-cells
py::array_t<int> py_ecp_2d3c_hl(py::array_t<int> image_c1,
                                py::array_t<int> image_c2,
                                py::array_t<int> image_c3, int T1, int T2,
                                int T3) {
    auto img_c1 = image_c1.unchecked<2>();
    auto img_c2 = image_c2.unchecked<2>();
    auto img_c3 = image_c3.unchecked<2>();

    size_t num_rows = img_c1.shape(0);
    size_t num_cols = img_c2.shape(1);

    Matrix<int, 3> ecp(T1 + 1, T2 + 1, T3 + 1, 0);

    // inner pixels
    for (size_t i = 1; i < num_rows - 1; ++i) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            // top-right vertex
            ecp(std::min({img_c1(i, j), img_c1(i - 1, j), img_c1(i - 1, j + 1),
                          img_c1(i, j + 1)}),
                std::min({img_c2(i, j), img_c2(i - 1, j), img_c2(i - 1, j + 1),
                          img_c2(i, j + 1)}),
                std::min({img_c3(i, j), img_c3(i - 1, j), img_c3(i - 1, j + 1),
                          img_c3(i, j + 1)})) += 1;

            // top edge
            ecp(std::min({img_c1(i, j), img_c1(i - 1, j)}),
                std::min({img_c2(i, j), img_c2(i - 1, j)}),
                std::min({img_c3(i, j), img_c3(i - 1, j)})) += -1;

            // right edge{
            ecp(std::min({img_c1(i, j), img_c1(i, j + 1)}),
                std::min({img_c2(i, j), img_c2(i, j + 1)}),
                std::min({img_c3(i, j), img_c3(i, j + 1)})) += -1;

            // square
            ecp(img_c1(i, j), img_c2(i, j), img_c3(i, j)) += 1;
        }
    }

    // top pixels
    // for(size_t j=1; j<num_cols-1; ++j) {
    // top-right vertex - is canceled by the right edge
    // ecp(std::min(img_c1(0,j), img_c1(0,j+1)),
    //     std::min(img_c2(0,j), img_c2(0,j+1)),
    //     std::min(img_c3(0,j), img_c3(0,j+1))) += 1;

    // top edge - is canceled by the square
    // ecp(img_c1(0,j), img_c2(0,j), img_c3(0,j)) += -1;

    // right edge - is canceled by the top-right vertex
    // ecp(std::min(img_c1(0,j), img_c1(0,j+1)),
    //     std::min(img_c1(0,j), img_c1(0,j+1)),
    //     std::min(img_c1(0,j), img_c1(0,j+1))) += -1;

    // square - is canceled by the top edge
    // ecp(img_c1(0,j), img_c2(0,j), img_c3(0,j)) += 1;
    // }

    // right pixels
    // for(size_t i=1; i<num_rows-1; ++i) {
    // top-right vertex - is canceled by the top edge
    // top edge - is canceled by the top-right vertex
    // right edge - is canceled by the square
    // square - is canceled by the right edge
    // }

    // bottom pixels
    for (size_t j = 1; j < num_cols - 1; ++j) {
        // top-right vertex
        ecp(std::min({img_c1(num_rows - 1, j), img_c1(num_rows - 2, j),
                      img_c1(num_rows - 2, j + 1),
                      img_c1(num_rows - 1, j + 1)}),
            std::min({img_c2(num_rows - 1, j), img_c2(num_rows - 2, j),
                      img_c2(num_rows - 2, j + 1),
                      img_c2(num_rows - 1, j + 1)}),
            std::min({img_c3(num_rows - 1, j), img_c3(num_rows - 2, j),
                      img_c3(num_rows - 2, j + 1),
                      img_c3(num_rows - 1, j + 1)})) += 1;

        // bottom-right vertex - is canceled by the right edge

        // top edge
        ecp(std::min({img_c1(num_rows - 1, j), img_c1(num_rows - 2, j)}),
            std::min({img_c2(num_rows - 1, j), img_c2(num_rows - 2, j)}),
            std::min({img_c3(num_rows - 1, j), img_c3(num_rows - 2, j)})) += -1;

        // right edge - is canceled by the bottom-right vertex
        // bottom edge - is canceled by the square
        // square - is canceled by the bottom edge
    }

    // left pixels
    for (size_t i = 1; i < num_rows - 1; ++i) {
        // top-right vertex
        ecp(std::min({img_c1(i, 0), img_c1(i - 1, 0), img_c1(i - 1, 1),
                      img_c1(i, 1)}),
            std::min({img_c2(i, 0), img_c2(i - 1, 0), img_c2(i - 1, 1),
                      img_c2(i, 1)}),
            std::min({img_c3(i, 0), img_c3(i - 1, 0), img_c3(i - 1, 1),
                      img_c3(i, 1)})) += 1;

        // top-left vertex - is canceled by the top edge

        // top edge - is canceled by the top-left vertex

        // right edge
        ecp(std::min({img_c1(i, 0), img_c1(i, 1)}),
            std::min({img_c2(i, 0), img_c2(i, 1)}),
            std::min({img_c3(i, 0), img_c3(i, 1)})) += -1;

        // left edge - is canceled by the square

        // square - is canceled by the left edge
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

    // bottom-left pixel
    // top-right vertex
    ecp(std::min({img_c1(num_rows - 1, 0), img_c1(num_rows - 2, 0),
                  img_c1(num_rows - 2, 1), img_c1(num_rows - 1, 1)}),
        std::min({img_c2(num_rows - 1, 0), img_c2(num_rows - 2, 0),
                  img_c2(num_rows - 2, 1), img_c2(num_rows - 1, 1)}),
        std::min({img_c3(num_rows - 1, 0), img_c3(num_rows - 2, 0),
                  img_c3(num_rows - 2, 1), img_c3(num_rows - 1, 1)})) += 1;
    // top-left vertex - is canceled by the top edge
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

py::array_t<int> py_ecp_2d3c_hale_v(py::array_t<int> image_c1,
                                    py::array_t<int> image_c2,
                                    py::array_t<int> image_c3, int T1, int T2,
                                    int T3) {
    auto img_c1 = image_c1.unchecked<2>();
    auto img_c2 = image_c2.unchecked<2>();
    auto img_c3 = image_c3.unchecked<2>();

    size_t num_rows = img_c1.shape(0);
    size_t num_cols = img_c2.shape(1);

    Matrix<int, 3> ecp(T1 + 1, T2 + 1, T3 + 1, 0);

    // o----
    // |
    // |

    // inner pixels
    for (size_t i = 0; i < num_rows - 1; ++i) {
        for (size_t j = 0; j < num_cols - 1; ++j) {
            // square
            ecp(std::max({img_c1(i, j), img_c1(i + 1, j), img_c1(i + 1, j + 1),
                          img_c1(i, j + 1)}),
                std::max({img_c2(i, j), img_c2(i + 1, j), img_c2(i + 1, j + 1),
                          img_c2(i, j + 1)}),
                std::max({img_c3(i, j), img_c3(i + 1, j), img_c3(i + 1, j + 1),
                          img_c3(i, j + 1)})) += 1;

            // top edge
            ecp(std::max({img_c1(i, j), img_c1(i, j + 1)}),
                std::max({img_c2(i, j), img_c2(i, j + 1)}),
                std::max({img_c3(i, j), img_c3(i, j + 1)})) += -1;

            // left edge
            ecp(std::max({img_c1(i, j), img_c1(i + 1, j)}),
                std::max({img_c2(i, j), img_c2(i + 1, j)}),
                std::max({img_c3(i, j), img_c3(i + 1, j)})) += -1;

            // top-left vertex
            ecp(img_c1(i, j), img_c2(i, j), img_c3(i, j)) += 1;
        }
    }

    // pixels at the right edge
    for (size_t i = 0; i < num_rows - 1; ++i) {
        // top-left vertex
        ecp(img_c1(i, num_cols - 1), img_c2(i, num_cols - 1),
            img_c3(i, num_cols - 1)) += 1;

        // left edge
        ecp(std::max({img_c1(i, num_cols - 1), img_c1(i + 1, num_cols - 1)}),
            std::max({img_c2(i, num_cols - 1), img_c2(i + 1, num_cols - 1)}),
            std::max({img_c3(i, num_cols - 1), img_c3(i + 1, num_cols - 1)})) +=
            -1;
    }

    // pixels at the bottom edge
    for (size_t j = 0; j < num_cols - 1; ++j) {
        // top-left vertex
        ecp(img_c1(num_rows - 1, j), img_c2(num_rows - 1, j),
            img_c3(num_rows - 1, j)) += 1;

        // top edge
        ecp(std::max({img_c1(num_rows - 1, j), img_c1(num_rows - 1, j + 1)}),
            std::max({img_c2(num_rows - 1, j), img_c2(num_rows - 1, j + 1)}),
            std::max({img_c3(num_rows - 1, j), img_c3(num_rows - 1, j + 1)})) +=
            -1;
    }

    // bottom-right pixel
    ecp(img_c1(num_rows - 1, num_cols - 1), img_c2(num_rows - 1, num_cols - 1),
        img_c3(num_rows - 1, num_cols - 1)) += 1;

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

py::array_t<int> compute_contributions(py::array_t<int> image_c1,
                                       py::array_t<int> image_c2,
                                       py::array_t<int> image_c3) {
    auto img_c1 = image_c1.unchecked<2>();
    auto img_c2 = image_c2.unchecked<2>();
    auto img_c3 = image_c3.unchecked<2>();

    size_t num_rows = img_c1.shape(0);
    size_t num_cols = img_c2.shape(1);

    // Matrix:
    // [ dimension-sign, R, G, B,
    // dimension-sign, R, G, B, ... ]
    size_t num_cubes = (2 * num_rows + 1) * (2 * num_cols + 1);
    Matrix<int, 2> contributions(num_cubes, 4, 0);

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

py::array_t<int> py_ecp_2d3c_hl_contributions(py::array_t<int> contributions,
                                              int T1, int T2, int T3) {
    auto contribs = contributions.unchecked<2>();
    size_t num_cubes = contribs.shape(0);

    Matrix<int, 3> ecp(T1 + 1, T2 + 1, T3 + 1, 0);

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

int pwr_sgn(int n) { return 1 - 2 * (1 & n); }

// only every called for vertices => dim = 1 => dim + |A| - 1 = |A|
void place_markers(Matrix<int, 3> &ecp, std::initializer_list<int> channel1,
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

py::array_t<int> py_ecp_2d3c_hl_mc(py::array_t<int> image_c1,
                                   py::array_t<int> image_c2,
                                   py::array_t<int> image_c3, int T1, int T2,
                                   int T3) {
    auto img_c1 = image_c1.unchecked<2>();
    auto img_c2 = image_c2.unchecked<2>();
    auto img_c3 = image_c3.unchecked<2>();

    size_t num_rows = img_c1.shape(0);
    size_t num_cols = img_c2.shape(1);

    Matrix<int, 3> ecp(T1 + 1, T2 + 1, T3 + 1, 0);

    // inner pixels
    for (size_t i = 1; i < num_rows - 1; ++i) {
        for (size_t j = 1; j < num_cols - 1; ++j) {
            // top-right vertex
            place_markers(ecp,
                          {img_c1(i, j), img_c1(i - 1, j), img_c1(i - 1, j + 1),
                           img_c1(i, j + 1)},
                          {img_c2(i, j), img_c2(i - 1, j), img_c2(i - 1, j + 1),
                           img_c2(i, j + 1)},
                          {img_c3(i, j), img_c3(i - 1, j), img_c3(i - 1, j + 1),
                           img_c3(i, j + 1)},
                          0);

            // top edge
            place_markers(ecp, {img_c1(i, j), img_c1(i - 1, j)},
                          {img_c2(i, j), img_c2(i - 1, j)},
                          {img_c3(i, j), img_c3(i - 1, j)}, 1);

            // right edge{
            place_markers(ecp, {img_c1(i, j), img_c1(i, j + 1)},
                          {img_c2(i, j), img_c2(i, j + 1)},
                          {img_c3(i, j), img_c3(i, j + 1)}, 1);

            // square
            ecp(img_c1(i, j), img_c2(i, j), img_c3(i, j)) += 1;
        }
    }

    // top pixels
    // for(size_t j=1; j<num_cols-1; ++j) {
    // top-right vertex - is canceled by the right edge
    // ecp(std::min(img_c1(0,j), img_c1(0,j+1)),
    //     std::min(img_c2(0,j), img_c2(0,j+1)),
    //     std::min(img_c3(0,j), img_c3(0,j+1))) += 1;

    // top edge - is canceled by the square
    // ecp(img_c1(0,j), img_c2(0,j), img_c3(0,j)) += -1;

    // right edge - is canceled by the top-right vertex
    // ecp(std::min(img_c1(0,j), img_c1(0,j+1)),
    //     std::min(img_c1(0,j), img_c1(0,j+1)),
    //     std::min(img_c1(0,j), img_c1(0,j+1))) += -1;

    // square - is canceled by the top edge
    // ecp(img_c1(0,j), img_c2(0,j), img_c3(0,j)) += 1;
    // }

    // right pixels
    // for(size_t i=1; i<num_rows-1; ++i) {
    // top-right vertex - is canceled by the top edge
    // top edge - is canceled by the top-right vertex
    // right edge - is canceled by the square
    // square - is canceled by the right edge
    // }

    // bottom pixels
    for (size_t j = 1; j < num_cols - 1; ++j) {
        // top-right vertex
        place_markers(
            ecp,
            {img_c1(num_rows - 1, j), img_c1(num_rows - 2, j),
             img_c1(num_rows - 2, j + 1), img_c1(num_rows - 1, j + 1)},
            {img_c2(num_rows - 1, j), img_c2(num_rows - 2, j),
             img_c2(num_rows - 2, j + 1), img_c2(num_rows - 1, j + 1)},
            {img_c3(num_rows - 1, j), img_c3(num_rows - 2, j),
             img_c3(num_rows - 2, j + 1), img_c3(num_rows - 1, j + 1)},
            0);

        // bottom-right vertex - is canceled by the right edge

        // top edge
        place_markers(ecp, {img_c1(num_rows - 1, j), img_c1(num_rows - 2, j)},
                      {img_c2(num_rows - 1, j), img_c2(num_rows - 2, j)},
                      {img_c3(num_rows - 1, j), img_c3(num_rows - 2, j)}, 1);

        // right edge - is canceled by the bottom-right vertex
        // bottom edge - is canceled by the square
        // square - is canceled by the bottom edge
    }

    // left pixels
    for (size_t i = 1; i < num_rows - 1; ++i) {
        // top-right vertex
        place_markers(
            ecp,
            {img_c1(i, 0), img_c1(i - 1, 0), img_c1(i - 1, 1), img_c1(i, 1)},
            {img_c2(i, 0), img_c2(i - 1, 0), img_c2(i - 1, 1), img_c2(i, 1)},
            {img_c3(i, 0), img_c3(i - 1, 0), img_c3(i - 1, 1), img_c3(i, 1)},
            0);

        // top-left vertex - is canceled by the top edge

        // top edge - is canceled by the top-left vertex

        // right edge
        place_markers(ecp, {img_c1(i, 0), img_c1(i, 1)},
                      {img_c2(i, 0), img_c2(i, 1)},
                      {img_c3(i, 0), img_c3(i, 1)}, 1);

        // left edge - is canceled by the square

        // square - is canceled by the left edge
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

    // bottom-left pixel
    // top-right vertex
    place_markers(ecp,
                  {img_c1(num_rows - 1, 0), img_c1(num_rows - 2, 0),
                   img_c1(num_rows - 2, 1), img_c1(num_rows - 1, 1)},
                  {img_c2(num_rows - 1, 0), img_c2(num_rows - 2, 0),
                   img_c2(num_rows - 2, 1), img_c2(num_rows - 1, 1)},
                  {img_c3(num_rows - 1, 0), img_c3(num_rows - 2, 0),
                   img_c3(num_rows - 2, 1), img_c3(num_rows - 1, 1)},
                  0);
    // top-left vertex - is canceled by the top edge
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

PYBIND11_MODULE(euprima, m) {
    m.def(
        "ec_binary_image_2d_naive", &py_ec_binary_image_2d_naive,
        "Euler characteristic of a binary 2d image without and optimizations");
    m.def(
        "char_binary_image_2d", &char_binary_image_2d,
        "Euler characteristic of a binary 2d image without and optimizations");
    m.def("ec_binary_image_2d_gray", &py_ec_binary_image_2d_gray,
          "Euler characteristic of a binary 2d image with Grays method");
    m.def("ec_binary_image_2d_yao", &py_ec_binary_image_2d_yao,
          "Euler characteristic of a binary 2d image with Yao's optimization");

    m.def("ecp_2d2c_naive", &py_ecp_2d2c_naive,
          "Euler characteristic profile computed without any optimizations");
    m.def("ecp_2d2c", &ecp_2d2c, "ECP of a two-dimensional two-channel image");

    m.def("euler_changes_2d", &py_euler_changes_2d,
          "Vector of all possible changes in Euler characteristic");

    m.def("ecp_2d3c_naive", &py_ecp_2d3c_naive2,
          "Euler characteristic profile computed without any optimizations");

    m.def("ecp_2d3c", &py_ecp_2d3c,
          "Euler characteristic profile computed with the most efficient "
          "algorithm yet");
    m.def("ecp_2d3c_optimized", &py_ecp_2d3c_optimized,
          "Euler characteristic profile computed with the most efficient "
          "algorithm yet");

    m.def("ecp_2d2c_hw", &py_ecp_2d2c_hw,
          "Euler characteristic profile computed the efficient Heiss/Wagner "
          "algorithm for ECCs");
    m.def("ecp_2d2c_hw_optimized", &py_ecp_2d2c_hw_optimized,
          "Euler characteristic profile computed the efficient Heiss/Wagner "
          "algorithm for ECCs");
    m.def("ecp_2d2c_hw_optimized2", &py_ecp_2d2c_hw_optimized2,
          "Euler characteristic profile computed the efficient Heiss/Wagner "
          "algorithm for ECCs");

    m.def("ecp_2d3c_hw_optimized", &py_ecp_2d3c_hw_optimized,
          "Euler characteristic profile computed the efficient Heiss/Wagner "
          "algorithm for ECCs");
    m.def("ecp_2d3c_hw_optimized_test", &py_ecp_2d3c_hw_optimized_test,
          "Euler characteristic profile computed the efficient Heiss/Wagner "
          "algorithm for ECCs");

    m.def("ecp_2d2c_optimized", &py_ecp_2d2c_optimized,
          "Euler characteristic profile computed the efficient Heiss/Wagner "
          "algorithm for ECCs");

    m.def("ecp_2d3c_hl", &py_ecp_2d3c_hl,
          "Euler characteristic profile for the top-cell filtration by "
          "Hacquard/Lebovici");
    m.def("ecp_2d3c_hl_mc", &py_ecp_2d3c_hl_mc,
          "Euler characteristic profile for the filtration induced by top-cell "
          "by Hacquard/Lebovici");
    m.def("ecp_2d3c_hl_contributions", &py_ecp_2d3c_hl_contributions,
          "Euler characteristic profile for the top-cell filtration by "
          "Hacquard/Lebovici");

    m.def("ecp_2d3c_hale_v", &py_ecp_2d3c_hale_v,
          "Euler characteristic profile for the vertex construction");

    m.def("compute_contributions", &compute_contributions,
          "Dlotko/Gurnari contributions / vectorized cubical complex for "
          "eulearning");
}
