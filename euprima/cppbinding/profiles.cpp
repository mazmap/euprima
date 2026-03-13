#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <vector>
#include <array>
#include <stdexcept>
#include <bitset>

#include <iostream>
#include <iomanip>

template <typename T, size_t N>
class Matrix {
private:
    std::vector<T> data;
    std::vector<size_t> shape;
    std::array<size_t, N> strides; // possible optimization by using std::array instead

    size_t idx(const std::vector<size_t> &indices) const {
	size_t idx = 0;
	for(size_t i=0; i<indices.size(); ++i) {
	    idx += indices[i] * strides[i];
	}
	return idx;
    }
public:
    Matrix(std::initializer_list<size_t> dims, T init_val = T())
	: shape(dims), strides() {
	
	size_t total_size = 1;
	for(auto d : shape) total_size *=d;
	data.assign(total_size, init_val);

	// Strides for row-major order
	size_t current_stride = 1;
	for(int i=shape.size()-1; i>=0; --i) {
	    strides[i] = current_stride;
	    current_stride *= shape[i];
	}
    }

    template<typename... Args>
    T& operator()(Args... args) {
	static_assert(sizeof...(Args) > 0, "Matrix requires at least one index");

	size_t idx = 0;
	size_t dim = 0;

	// "comma-fold" suggested by Gemini
	([&] {
	    idx += static_cast<size_t>(args)*strides[dim++];
	 } (), ...);

	return data[idx];
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
	static_assert(sizeof...(Args) > 0, "Matrix requires at least one index");

	size_t idx = 0;
	size_t dim = 0;

	// "comma-fold" suggested by Gemini
	([&] {
	    idx += static_cast<size_t>(args)*strides[dim++];
	 } (), ...);

	return data[idx];
    }

    T& operator()(size_t r, size_t c) { return data[r*strides[0] + c*strides[1]]; }

    size_t dim(size_t index) {
	return shape[index];
    }

    const size_t dim(size_t index) const {
	return shape[index];
    }

    T* data_ptr() {
	return data.data();
    }
};

template <typename T>
class Matrix<T, 2> {
    size_t rows, cols;
    std::vector<T> data;
public:
    Matrix(size_t r, size_t c, T init_val = T()) : rows(r), cols(c), data(r * c, init_val) {}

    // No folds, no lambdas. Pure, simple math the compiler loves.
    T& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }
    
    const T& operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    size_t dim(size_t index) {
	return index == 0 ? rows : cols;
    }

    const size_t dim(size_t index) const {
	return index == 0 ? rows : cols;
    }

    T* data_ptr() {
	return data.data();
    }

};

template <typename T>
class Matrix<T, 3> {
    size_t dim1, dim2, dim3;
    std::vector<T> data;
public:
    Matrix(size_t d1, size_t d2, size_t d3, T init_val = T()) : dim1(d1), dim2(d2), dim3(d3), data(d1*d2*d3, init_val) {}

    // No folds, no lambdas. Pure, simple math the compiler loves.
    T& operator()(size_t i, size_t j, size_t k) {
        return data[i*dim2*dim3 + j*dim3 + k];
    }
    
    const T& operator()(size_t i, size_t j, size_t k) const {
        return data[i*dim2*dim3 + j*dim3 + k];
    }

    size_t dim(size_t index) {
	if(index==0)
	    return dim1;
	else if(index==1)
	    return dim2;
	else
	    return dim3;
    }

    const size_t dim(size_t index) const {
	if(index==0)
	    return dim1;
	else if(index==1)
	    return dim2;
	else
	    return dim3;
    }

    T* data_ptr() {
	return data.data();
    }

};

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const Matrix<T, N>& mat) {
    for (size_t i = 0; i < mat.dim(0); ++i) {
        os << "[ ";
        for (size_t j = 0; j < mat.dim(1); ++j) {
            // Using setw(4) keeps columns aligned even with different digit counts
            os << std::setw(4) << mat(i, j) << " ";
        }
        os << "]\n";
    }
    return os;
}

template <size_t N>
std::bitset<N> reverse(std::bitset<N> b) {
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

    for(size_t j=1; j < num_cols-1; ++j) {
	// iterate through pixels at the top edge
	f += image(0,j);
	v += image(0,j) || image(0,j+1); // top-right vertex
	e += image(0,j); // upper edge
	e += image(0,j) || image(0,j+1); // right edge
    }

    for(size_t i=1; i < num_rows-1; ++i) {
	// iterate through pixels at right edge
	f += image(i,num_cols-1);
	v += image(i,num_cols-1) || image(i-1,num_cols-1); // top-right vertex
	e += image(i,num_cols-1) || image(i-1,num_cols-1); // upper edge
	e += image(i,num_cols-1); // right edge
    }

    for(size_t i=1; i < num_rows-1; ++i) {
	// iterate through pixels at left edge
	f += image(i,0); 

	v += image(i,0) || image(i-1,0); // top-left vertex
	e += image(i,0); // left edge
	
	v += image(i,0) || image(i-1,0) || image(i-1,1) || image(i,1); // upper right vertex
	e += image(i,0) || image(i-1,0); // upper edge
	e += image(i,0) || image(i,1); // right edge
    }

    for(size_t j=1; j < num_cols-1; ++j) {
	// iterate through pixels at the bottom edge
	f += image(num_rows-1,j);

	v += image(num_rows-1,j) || image(num_rows-1,j+1); // bottom-right vertex
	e += image(num_rows-1,j); // bottom edge
	
	v += image(num_rows-1,j) || image(num_rows-2,j) || image(num_rows-2,j+1) || image(num_rows-1,j+1); // upper right vertex
	e += image(num_rows-1,j) || image(num_rows-2,j); // upper edge
	e += image(num_rows-1,j) || image(num_rows-1,j+1); // right edge
    }

    // top-left pixel
    f += image(0,0);
    v += image(0,0) || image(0,1); // top-right vertex
    e += 2*image(0,0); // upper edge + left edge
    e += image(0,0) || image(0,1); // right edge
    v += image(0,0); // top-left vertex

    // top-right pixel
    f += image(0,num_cols-1);
    v += image(0,num_cols-1); // top-right vertex
    e += 2*image(0,num_cols-1); // top edge + right edge

    // bottom-left pixel
    f += image(num_rows-1,0);
    v += image(num_rows-1,0); // bottom-left vertex
    e += 2*image(num_rows-1,0); // left edge + bottom edge
    v += image(num_rows-1,0) || image(num_rows-2,0); // top-left vertex
    v += image(num_rows-1,0) || image(num_rows-1,1); // bottom-right vertex
    v += image(num_rows-1,0) || image(num_rows-2,0) || image(num_rows-2,1) || image(num_rows-1,1); // top-right vertex
    e += image(num_rows-1,0) || image(num_rows-2,0); // top edge
    e += image(num_rows-1,0) || image(num_rows-1,1); // right edge

    // bottom-right pixel
    f += image(num_rows-1,num_cols-1);
    v += image(num_rows-1,num_cols-1); // bottom-right vertex
    e += 2*image(num_rows-1,num_cols-1); // bottom edge + right edge
    e += image(num_rows-1,num_cols-1) || image(num_rows-2,num_cols-1); // upper edge
    v += image(num_rows-1,num_cols-1) || image(num_rows-2,num_cols-1); // top-right vertex

    for(size_t i = 1; i < num_rows-1; ++i) {
	for(size_t j = 1; j < num_cols-1; ++j) {
	    f += image(i,j);

	    v += image(i,j) || image(i-1,j) || image(i-1,j+1) || image(i,j+1); // top-right vertex
	    e += image(i,j) || image(i-1,j); // top edge
	    e += image(i,j) || image(i,j+1); // right edge
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

    for(size_t j=1; j < num_cols-1; ++j) {
	// iterate through pixels at the top edge
	f += img(0,j);
	v += img(0,j) || img(0,j+1); // top-right vertex
	e += img(0,j); // upper edge
	e += img(0,j) || img(0,j+1); // right edge
    }

    for(size_t i=1; i < num_rows-1; ++i) {
	// iterate through pixels at right edge
	f += img(i,num_cols-1);
	v += img(i,num_cols-1) || img(i-1,num_cols-1); // top-right vertex
	e += img(i,num_cols-1) || img(i-1,num_cols-1); // upper edge
	e += img(i,num_cols-1); // right edge
    }

    for(size_t i=1; i < num_rows-1; ++i) {
	// iterate through pixels at left edge
	f += img(i,0); 

	v += img(i,0) || img(i-1,0); // top-left vertex
	e += img(i,0); // left edge
	
	v += img(i,0) || img(i-1,0) || img(i-1,1) || img(i,1); // upper right vertex
	e += img(i,0) || img(i-1,0); // upper edge
	e += img(i,0) || img(i,1); // right edge
    }

    for(size_t j=1; j < num_cols-1; ++j) {
	// iterate through pixels at the bottom edge
	f += img(num_rows-1,j);

	v += img(num_rows-1,j) || img(num_rows-1,j+1); // bottom-right vertex
	e += img(num_rows-1,j); // bottom edge
	
	v += img(num_rows-1,j) || img(num_rows-2,j) || img(num_rows-2,j+1) || img(num_rows-1,j+1); // upper right vertex
	e += img(num_rows-1,j) || img(num_rows-2,j); // upper edge
	e += img(num_rows-1,j) || img(num_rows-1,j+1); // right edge
    }

    // top-left pixel
    f += img(0,0);
    v += img(0,0) || img(0,1); // top-right vertex
    e += 2*img(0,0); // upper edge + left edge
    e += img(0,0) || img(0,1); // right edge
    v += img(0,0); // top-left vertex

    // top-right pixel
    f += img(0,num_cols-1);
    v += img(0,num_cols-1); // top-right vertex
    e += 2*img(0,num_cols-1); // top edge + right edge

    // bottom-left pixel
    f += img(num_rows-1,0);
    v += img(num_rows-1,0); // bottom-left vertex
    e += 2*img(num_rows-1,0); // left edge + bottom edge
    v += img(num_rows-1,0) || img(num_rows-2,0); // top-left vertex
    v += img(num_rows-1,0) || img(num_rows-1,1); // bottom-right vertex
    v += img(num_rows-1,0) || img(num_rows-2,0) || img(num_rows-2,1) || img(num_rows-1,1); // top-right vertex
    e += img(num_rows-1,0) || img(num_rows-2,0); // top edge
    e += img(num_rows-1,0) || img(num_rows-1,1); // right edge

    // bottom-right pixel
    f += img(num_rows-1,num_cols-1);
    v += img(num_rows-1,num_cols-1); // bottom-right vertex
    e += 2*img(num_rows-1,num_cols-1); // bottom edge + right edge
    e += img(num_rows-1,num_cols-1) || img(num_rows-2,num_cols-1); // upper edge
    v += img(num_rows-1,num_cols-1) || img(num_rows-2,num_cols-1); // top-right vertex

    for(size_t i = 1; i < num_rows-1; ++i) {
	for(size_t j = 1; j < num_cols-1; ++j) {
	    f += img(i,j);

	    v += img(i,j) || img(i-1,j) || img(i-1,j+1) || img(i,j+1); // top-right vertex
	    e += img(i,j) || img(i-1,j); // top edge
	    e += img(i,j) || img(i,j+1); // right edge
	}
    }

    return v - e + f;
}

int sum_bool_2d(Matrix<uint8_t,2> &matrix)
{
    size_t numI = matrix.dim(0);
    size_t numJ = matrix.dim(1);

    int total = 0;
    for (size_t i = 0; i < numI; ++i) {
        for (size_t j = 0; j < numJ; ++j) {
            if (matrix(i,j) == 1)
                total += 1;
        }
    }

    return total;
}

int sum_bool_2d(py::array_t<uint8_t> matrix)
{
    auto img = matrix.unchecked<2>();

    size_t numI = matrix.shape(0);
    size_t numJ = matrix.shape(1);

    int total = 0;
    for (size_t i = 0; i < numI; ++i) {
        for (size_t j = 0; j < numJ; ++j) {
            if (img(i,j) == 1)
                total += 1;
        }
    }

    return total;
}

int char_binary_image_2d(py::array_t<uint8_t> input)
{
    auto img = input.unchecked<2>();

    // Input shape: binary image number of rows and columns
    size_t numI = img.shape(0);
    size_t numJ = img.shape(1);

    // Matrices for vectices, horizontal edges and vertical edges
    Matrix<uint8_t,2> V(numI+1,numJ+1,0);
    Matrix<uint8_t,2> Eh(numI+1,numJ,0);
    Matrix<uint8_t,2> Ev(numI, numJ+1,0);

    // Loop over pixels to update V, Eh, Ev
    for (size_t i = 0; i < numI; ++i) {
        for (size_t j = 0; j < numJ; ++j) {
            if (img(i,j) == 1) {
                V(i,j)     = 1;
                V(i+1,j)   = 1;
                V(i,j+1)   = 1;
                V(i+1,j+1) = 1;

                Eh(i,j)    = 1;
                Eh(i+1,j)  = 1;

                Ev(i,j)    = 1;
                Ev(i,j+1)  = 1;
            }
        }
    }

    // Sum of elements in the matrices
    int v  = sum_bool_2d(V);
    int eh = sum_bool_2d(Eh);
    int ev = sum_bool_2d(Ev);
    int f  = sum_bool_2d(input);

    int EC = v - eh - ev + f;
    return EC;
}

int py_ec_binary_image_2d_gray(py::array_t<uint8_t> image) {
    auto img = image.unchecked<2>();

    size_t num_rows = img.shape(0);
    size_t num_cols = img.shape(1);

    int q1 = 0, q3 = 0, qd = 0;

    // top row
    for(size_t j=0; j < num_rows-1; ++j) {
	int a = img(0,j);
	int b = img(0,j+1);
	q1 += (a && !b) || (!a && b);
    }
    // right row 
    for(size_t i=0; i < num_cols-1; ++i) {
	int a = img(i,0);
	int b = img(i+1,0);
	q1 += (a && !b) || (!a && b);
    }
    // bottom row 
    for(size_t j=0; j < num_rows-1; ++j) {
	int a = img(num_rows-1,j);
	int b = img(num_rows-1,j+1);
	q1 += (a && !b) || (!a && b);
    }
    // left row
    for(size_t i=0; i < num_cols-1; ++i) {
	int a = img(i,num_cols-1);
	int b = img(i+1,num_cols-1);
	q1 += (a && !b) || (!a && b);
    }
    // top-left bit quad 
    q1 += img(0,0);
    // top-right bit quad 
    q1 += img(0,num_cols-1);
    // bottom-left bit quad 
    q1 += img(num_rows-1,0);
    // bottom-right bit quad
    q1 += img(num_rows-1,num_cols-1);

    for(size_t i=0; i < num_rows-1; ++i) {
	for(size_t j=0; j < num_cols-1; ++j) {
	    uint8_t a = img(i,j);
	    uint8_t b = img(i,j+1);
	    uint8_t c = img(i+1,j);
	    uint8_t d = img(i+1,j+1);

	    uint8_t count = a + b + c + d;

	    if(count == 1) 
		q1++;
	    else if(count == 3)
		q3++;
	    else if(count == 2 && a == d) 
		qd++;
	}
    }

    return (q1 - q3 - 2*qd) / 4; 
}

// does not work...
int py_ec_binary_image_2d_yao(py::array_t<uint8_t> image) {
    auto uimg = image.unchecked<2>();

    size_t num_rows = uimg.shape(0);
    size_t num_cols = uimg.shape(1);

    Matrix<uint8_t,2> img(num_rows+2,num_cols+2,0);
    for(size_t i = 1; i < num_rows+1; ++i) {
	for(size_t j = 1; j < num_cols+1; ++j) {
	    img(i,j) = uimg(i-1,j-1);
	}
    }

    num_rows += 2;
    num_cols += 2;

    int w2=0, wc=0;
    size_t x=2, y=2;

    while(y <= num_cols) {
	while(x <= num_rows) {
	    if(img(x-2,y-2) == 1) 
		x++;
	    else if(img(x-2,y-1) == 0) {
labelA: 
		if(img(x-1,y-2) == 1) {
		    x += 2;
		} else if(img(x-1,y-1) == 0) {
		    x++;
		    if(x <= num_cols) 
			goto labelA;
		} else {
		    w2++;
		    x++;
		    if(x <= num_cols)
			goto labelB;
		}
	    } else {
labelB:
		if(img(x-1,y-2) == 1) {
		    wc++;
		    x += 2;
		} else if(img(x-1,y-1) == 0) {
		    x++;
		    if(x <= num_cols)
			goto labelA;
		} else {
		    x++;
		    if(x <= num_cols) 
			goto labelB;
		}
	    }
	}
	y++;
    }

    return w2 - wc;
}

Matrix<uint8_t, 2> py_binary_threshold_image_2d(py::array_t<int> image, int val) {
    auto img = image.unchecked<2>();

    size_t num_rows = img.shape(0);
    size_t num_cols = img.shape(1);

    Matrix<uint8_t, 2> thresholded_img(num_rows,num_cols,0);

    for(size_t i=0; i<num_rows; i++) {
	for(size_t j=0; j<num_cols; j++) {
	    thresholded_img(i,j) = img(i,j) <= val;
	}
    }

    return thresholded_img;
}

// ATTENTION: The following specification is not safe at all as Args accepts ANY type, not just matrices!
template<typename... Args>
Matrix<uint8_t,2> elementwise_AND_2d(const Matrix<uint8_t,2> &image1, const Args&... images) {
    Matrix<uint8_t,2> result = image1;

    size_t num_rows = image1.dim(0);
    size_t num_cols = image1.dim(1);

    for(size_t i=0; i<num_rows; i++){
	for(size_t j=0; j<num_cols; j++) {
	    result(i,j) = (result(i,j) && ... && images(i,j));
	}
    }

    return result;
}

Matrix<uint8_t,2> elementwise_AND_2d(const Matrix<uint8_t,2> &image1, const Matrix<uint8_t,2> &image2) {
    Matrix<uint8_t,2> result = image1;

    size_t num_rows = image1.dim(0);
    size_t num_cols = image1.dim(1);

    for(size_t i=0; i<num_rows; i++){
	for(size_t j=0; j<num_cols; j++) {
	    result(i,j) = (result(i,j) && image2(i,j));
	}
    }

    return result;
}

py::array_t<int> py_ecp_2d2c_naive(py::array_t<int> image_c1, py::array_t<int> image_c2, int T1, int T2) {
    Matrix<int,2> ecp(static_cast<size_t>(T1+1),static_cast<size_t>(T2+1),0);

    for(int i=0; i<=T1; i++) {
	Matrix<uint8_t,2> thresholded_img_c1 = py_binary_threshold_image_2d(image_c1, i);

	for(int j=0; j<=T2; j++) {
	    Matrix<uint8_t,2> thresholded_img_c2 = py_binary_threshold_image_2d(image_c2, j);
	    Matrix<uint8_t,2> Kij = elementwise_AND_2d(thresholded_img_c1, thresholded_img_c2);
	    ecp(i,j) = ec_binary_image_2d_naive(Kij);
	}
    }

    return py::array_t<int>(
	{ecp.dim(0), ecp.dim(1)},
	ecp.data_ptr()
    );
}

std::bitset<8> neigh_matrix(Matrix<int,2> &image, int i, int j, int f) {
    std::bitset<8> neigh;
    
    for(size_t p=0; p<3; ++p){
	for(size_t q=0; q<3; ++q){
	    if(q+3*p < 4) {
		neigh[q + 3*p] = image(i-1+p,j-1+q) <= f;
	    } else if(q+3*p > 4) {
		neigh[q + 3*p - 1] = image(i-1+p,j-1+q) <= f;
	    }
	}
    }

    return neigh;
}

std::bitset<8> neigh_matrix_lex(Matrix<int,2> &image, int i, int j, int f) {
    std::bitset<8> neigh;

    for(size_t p=0; p<3; ++p){
	for(size_t q=0; q<3; ++q){
	    if(q+3*p < 4) {
		neigh[q + 3*p] = image(i-1+p,j-1+q) <= f;
		// test(p,q) = image(i-1+p,j-1+q) <= f;
	    } else if(q+3*p > 4) {
		neigh[q + 3*p - 1] = image(i-1+p,j-1+q) <= f-1;
		// test(p,q) = image(i-1+p,j-1+q) <= f-1;
	    }
	    
	}
    }

    return neigh;
}

Matrix<int,2> pad_img_2d(py::array_t<int> image, int val) {
    auto img = image.unchecked<2>();

    size_t N = img.shape(0);
    size_t M = img.shape(1);

    Matrix<int,2> padded_img(N+2, M+2, val);

    for(size_t i=1; i < N+1; ++i) {
	for(size_t j=1; j<M+1; ++j) {
	    padded_img(i,j) = img(i-1,j-1);
	}
    }

    return padded_img;
}

py::array_t<int> ecp_2d2c(py::array_t<int> image_c1, py::array_t<int> image_c2, py::array_t<int> euler_changes, int T1, int T2) {
    auto ec = euler_changes.unchecked<1>();

    size_t N = image_c1.shape(0);
    size_t M = image_c1.shape(1);

    Matrix<int,2> ecp(static_cast<size_t>(T1+1), static_cast<size_t>(T2+1), 0);

    Matrix<int,2> padded_img_c1 = pad_img_2d(image_c1, T1+1);
    Matrix<int,2> padded_img_c2 = pad_img_2d(image_c2, T2+1);

    for(size_t i=1; i<N+1; ++i) {
	for(size_t j=1; j<M+1; ++j) {
	    int r = padded_img_c1(i,j); 
	    std::bitset<8> binary_neigh1 = neigh_matrix_lex(padded_img_c1, i, j, r);

	    std::vector<int> thresholds2(10, T2+1); // can be an array
	    int s = padded_img_c2(i,j);
	    for(size_t p=0; p<3; ++p) {
		for(size_t q=0; q<3; ++q) {
		    int val = padded_img_c2(i-1+p,j-1+q);
		    if(val >= s)
			thresholds2[q+p*3] = val;
		}
	    }

	    sort(thresholds2.begin(), thresholds2.end());
	    auto last = unique(thresholds2.begin(), thresholds2.end());
	    thresholds2.erase(last, thresholds2.end());

	    for(size_t z=0; z < thresholds2.size()-1; ++z) {
		size_t start = static_cast<size_t>(thresholds2[z]);
		size_t end = static_cast<size_t>(thresholds2[z+1]);

		std::bitset<8> binary_neigh2 = neigh_matrix(padded_img_c2, i, j, thresholds2[z]);
		std::bitset<8> binary_neigh = binary_neigh1 & binary_neigh2;

		size_t neigh_num = binary_neigh.to_ulong();
		int change = ec(neigh_num);

		for(size_t ind=start; ind < end; ++ind){
		    ecp(static_cast<int>(r),ind) += change; 
		}
	    }
	}
    }

    for(size_t s=1; s<T1+1; ++s){
	for(size_t t=0; t<T2+1; ++t) {
	    ecp(s,t) += ecp(s-1,t);
	}
    }

/*
    auto capsule = py::capsule(ecp, [](void *f) {
        delete reinterpret_cast<Matrix<int> *>(f);
    });
*/

    return py::array_t<int>(
        {ecp.dim(0), ecp.dim(1)},       // Shape
	ecp.data_ptr()
    );
}

int euler_change_neigh_matrix(Matrix<uint8_t,2> neigh) {
    uint8_t edges = neigh(0,1) + neigh(1,2) + neigh(2,1) + neigh(1,0);
    uint8_t vA = neigh(1,0) || neigh(0,0) || neigh(0,1);
    uint8_t vB = neigh(0,1) || neigh(0,2) || neigh(1,2);
    uint8_t vC = neigh(1,2) || neigh(2,2) || neigh(2,1);
    uint8_t vD = neigh(2,1) || neigh(2,0) || neigh(1,0);
    uint8_t vertices = vA + vB + vC + vD;
    return 1 - vertices + edges;
}

bool nth_bit(int number, int n) {
    return (number >> n) & 1;
}

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
    std::vector<int> euler_changes(256,0);

    for(int i=0; i<256; i++) {
	euler_changes[i] = euler_change_neigh_matrix(i);
    }

    return euler_changes;
}

py::array_t<int> py_euler_changes_2d() {
    std::vector<int> euler_changes = euler_changes_2d();

    return py::array_t<int>(
	256,
	euler_changes.data()
    );
}

py::array_t<int> py_ecp_2d3c_naive(py::array_t<int> image_c1, py::array_t<int> image_c2, py::array_t<int> image_c3, int T1, int T2, int T3) {
    Matrix<int,3> ecp(static_cast<size_t>(T1+1),static_cast<size_t>(T2+1),static_cast<size_t>(T3+1),0);

    for(int r=0; r<=T1; r++) {
	Matrix<uint8_t,2> thresholded_img_c1 = py_binary_threshold_image_2d(image_c1, r);

	for(int s=0; s<=T2; s++) {
	    Matrix<uint8_t,2> thresholded_img_c2 = py_binary_threshold_image_2d(image_c2, s);

	    for(int t=0; t<=T3; t++) {
		Matrix<uint8_t,2> thresholded_img_c3 = py_binary_threshold_image_2d(image_c3, t);
		Matrix<uint8_t,2> K_rst = elementwise_AND_2d(thresholded_img_c1, thresholded_img_c2, thresholded_img_c3);
		ecp(r,s,t) = ec_binary_image_2d_naive(K_rst);
	    }
	}
    }

    return py::array_t<int>(
	{ecp.dim(0), ecp.dim(1), ecp.dim(2)},
	ecp.data_ptr()
    );
}

py::array_t<int> py_ecp_2d3c(py::array_t<int> image_c1, py::array_t<int> image_c2, py::array_t<int> image_c3, py::array_t<int> euler_changes, int T1, int T2, int T3) {
    auto ec = euler_changes.unchecked<1>();

    size_t N = image_c1.shape(0);
    size_t M = image_c1.shape(1);

    Matrix<int,3> ecp(static_cast<size_t>(T1+1), static_cast<size_t>(T2+1), static_cast<size_t>(T3+1), 0);

    Matrix<int,2> padded_img_c1 = pad_img_2d(image_c1, T1+1);
    Matrix<int,2> padded_img_c2 = pad_img_2d(image_c2, T2+1);
    Matrix<int,2> padded_img_c3 = pad_img_2d(image_c3, T3+1);

    for(size_t i=1; i<N+1; ++i) {
	for(size_t j=1; j<M+1; ++j) {
	    int r = padded_img_c1(i,j); 
	    std::bitset<8> binary_neigh1 = neigh_matrix_lex(padded_img_c1, i, j, r);

	    std::vector<int> thresholds2(10, T2+1); // can be an array
	    int s = padded_img_c2(i,j);
	    for(size_t p=0; p<3; ++p) {
		for(size_t q=0; q<3; ++q) {
		    int val = padded_img_c2(i-1+p,j-1+q);
		    if(val >= s)
			thresholds2[q+p*3] = val;
		}
	    }
	    sort(thresholds2.begin(), thresholds2.end());
	    auto last = unique(thresholds2.begin(), thresholds2.end());
	    thresholds2.erase(last, thresholds2.end());

	    std::vector<int> thresholds3(10, T3+1);
	    int t = padded_img_c3(i,j);
	    for(size_t p=0; p<3; ++p) {
		for(size_t q=0; q<3; ++q) {
		    int val = padded_img_c3(i-1+p,j-1+q);
		    if(val >= t)
			thresholds3[q+p*3] = val;
		}
	    }
	    sort(thresholds3.begin(), thresholds3.end());
	    last = unique(thresholds3.begin(), thresholds3.end());
	    thresholds3.erase(last, thresholds3.end());

	    for(size_t z=0; z < thresholds2.size()-1; ++z) {
		size_t start2 = static_cast<size_t>(thresholds2[z]);
		size_t end2 = static_cast<size_t>(thresholds2[z+1]);

		std::bitset<8> binary_neigh2 = neigh_matrix(padded_img_c2, i, j, thresholds2[z]);

		for(size_t y=0; y<thresholds3.size()-1; ++y) {
		    size_t start3 = static_cast<size_t>(thresholds3[y]);
		    size_t end3 = static_cast<size_t>(thresholds3[y+1]);

		    std::bitset<8> binary_neigh3 = neigh_matrix(padded_img_c3, i, j, thresholds3[y]);
		    std::bitset<8> binary_neigh = binary_neigh1 & binary_neigh2 & binary_neigh3;

		    size_t neigh_num = binary_neigh.to_ulong();
		    int change = ec(neigh_num);

		    for(size_t ind2=start2; ind2 < end2; ++ind2){
			for(size_t ind3=start3; ind3 < end3; ++ind3) {
			    ecp(static_cast<int>(r),ind2,ind3) += change; 
			}
		    }
		}
		
	    }
	}
    }

    for(size_t r=1; r<T1+1; ++r){
	for(size_t s=0; s<T2+1; ++s) {
	    for(size_t t=0; t<T3+1; ++t) {
		ecp(r,s,t) += ecp(r-1,s,t);
	    }
	}
    }

/*
    auto capsule = py::capsule(ecp, [](void *f) {
        delete reinterpret_cast<Matrix<int> *>(f);
    });
*/

    return py::array_t<int>(
        {ecp.dim(0), ecp.dim(1), ecp.dim(2)},       // Shape
	ecp.data_ptr()
    );
}
// Heiss & Wagner
py::array_t<int> py_ecp_2d2c_hw(py::array_t<int> image_c1, py::array_t<int> image_c2, int T1, int T2) {
    Matrix<int,2> pad_img_c1 = pad_img_2d(image_c1,T1+1);
    Matrix<int,2> pad_img_c2 = pad_img_2d(image_c2,T2+1);

    size_t num_rows = image_c2.shape(0);
    size_t num_cols = image_c2.shape(1); 

    Matrix<int, 2> ecp(T1+1, T2+1, 0);

    for(size_t s=0; s<=T2; ++s) {
	for(size_t i=1; i<num_rows+1; ++i) {
	    for(size_t j=1; j<num_cols+1; ++j) {
		if(pad_img_c2(i,j) <= s) {
		    int val = pad_img_c1(i,j);

		    int n1 = pad_img_c2(i-1,j-1) <= s ? pad_img_c1(i-1,j-1) : T1+1;
		    int n2 = pad_img_c2(i-1,j) <= s ? pad_img_c1(i-1,j) : T1+1;
		    int n3 = pad_img_c2(i-1,j+1) <= s ? pad_img_c1(i-1,j+1) : T1+1;
		    int n4 = pad_img_c2(i,j-1) <= s ? pad_img_c1(i,j-1) : T1+1;
		    int n6 = pad_img_c2(i,j+1) <= s ? pad_img_c1(i,j+1) : T1+1;
		    int n7 = pad_img_c2(i+1,j-1) <= s ? pad_img_c1(i+1,j-1) : T1+1;
		    int n8 = pad_img_c2(i+1,j) <= s ? pad_img_c1(i+1,j) : T1+1;
		    int n9 = pad_img_c2(i+1,j+1) <= s ? pad_img_c1(i+1,j+1) : T1+1;

		    int change = 0;

		    // square
		    change += 1;

		    // edges
		    change -= n2 >= val; // top edge
		    change -= n6 > val; // right edge
		    change -= n8 > val; // bottom edge
		    change -= n4 >= val; // left edge
		    
		    // vertices
		    change += n1 >= val && n2 >= val && n4 >= val; // top-left vertex 
		    change += n2 >= val && n3 >= val && n6 > val; // top-right vertex 
		    change += n4 >= val && n7 > val && n8 > val; // bottom-left vertex 
		    change += n6 > val && n8 > val && n9  > val; // bottom-right vertex 

		    ecp(val,s) += change;
		}
	    }
	}

	// cummulative sum
	for(size_t r=1; r<=T1; ++r) {
	    ecp(r,s) += ecp(r-1,s);
	}
    }

    return py::array_t<int>(
	{ecp.dim(0), ecp.dim(1)},
	ecp.data_ptr()
    );
}

PYBIND11_MODULE(euprima, m) {
    m.def("ec_binary_image_2d_naive", &py_ec_binary_image_2d_naive, "Euler characteristic of a binary 2d image without and optimizations");
    m.def("char_binary_image_2d", &char_binary_image_2d, "Euler characteristic of a binary 2d image without and optimizations");
    m.def("ec_binary_image_2d_gray", &py_ec_binary_image_2d_gray, "Euler characteristic of a binary 2d image with Grays method");
    m.def("ec_binary_image_2d_yao", &py_ec_binary_image_2d_yao, "Euler characteristic of a binary 2d image with Yao's optimization");
    m.def("ecp_2d2c_naive", &py_ecp_2d2c_naive, "Euler characteristic profile computed without any optimizations");
    m.def("ecp_2d2c", &ecp_2d2c, "ECP of a two-dimensional two-channel image");
    m.def("euler_changes_2d", &py_euler_changes_2d, "Vector of all possible changes in Euler characteristic");
    m.def("ecp_2d3c_naive", &py_ecp_2d3c_naive, "Euler characteristic profile computed without any optimizations");
    m.def("ecp_2d3c", &py_ecp_2d3c, "Euler characteristic profile computed with the most efficient algorithm yet");
    m.def("ecp_2d2c_hw", &py_ecp_2d2c_hw, "Euler characteristic profile computed the efficient Heiss/Wagner algorithm for ECCs");
}
