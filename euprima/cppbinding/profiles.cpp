#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <vector>
#include <stdexcept>
#include <bitset>

#include <iostream>
#include <iomanip>

template <typename T>
class Matrix {
public:
    std::vector<T> data;
    size_t rows;
    size_t cols;

    Matrix(size_t r, size_t c, T init_val = 0) 
        : rows(r), cols(c), data(r * c, init_val) {}

    // Safe access with range checking
    T& at(size_t r, size_t c) {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data[r*cols + c];
    }

    // Fast access (no checks)
    T& operator()(size_t r, size_t c) { return data[r*cols + c]; }

    const T& operator()(size_t r, size_t c) const { return data[r*cols + c]; }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
    for (size_t i = 0; i < mat.rows; ++i) {
        os << "[ ";
        for (size_t j = 0; j < mat.cols; ++j) {
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
int ec_binary_image_2d_naive(Matrix<uint8_t> &image) {
    size_t num_rows = image.rows;
    size_t num_cols = image.cols;

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

int sum_bool_2d(Matrix<uint8_t> &matrix)
{
    size_t numI = matrix.rows;
    size_t numJ = matrix.cols;

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
    Matrix<uint8_t> V(numI+1,numJ+1,0);
    Matrix<uint8_t> Eh(numI+1,numJ,0);
    Matrix<uint8_t> Ev(numI, numJ+1,0);

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

    Matrix<uint8_t> img(num_rows+2,num_cols+2,0);
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

Matrix<uint8_t> py_binary_threshold_image_2d(py::array_t<int> image, int val) {
    auto img = image.unchecked<2>();

    size_t num_rows = img.shape(0);
    size_t num_cols = img.shape(1);

    Matrix<uint8_t> thresholded_img(num_rows,num_cols,0);

    for(size_t i=0; i<num_rows; i++) {
	for(size_t j=0; j<num_cols; j++) {
	    thresholded_img(i,j) = img(i,j) <= val;
	}
    }

    return thresholded_img;
}

Matrix<uint8_t> elementwise_AND_2d(Matrix<uint8_t> &image1, Matrix<uint8_t> &image2) {
    size_t num_rows = image1.rows;
    size_t num_cols = image1.cols;

    Matrix<uint8_t> result(num_rows, num_cols, 0);

    for(size_t i=0; i<num_rows; i++){
	for(size_t j=0; j<num_cols; j++) {
	    result(i,j) = image1(i,j) && image2(i,j);
	}
    }

    return result;
}

py::array_t<int> py_ecp_2d2c_naive(py::array_t<int> image_c1, py::array_t<int> image_c2, int T1, int T2) {
    Matrix<int> ecp(T1+1,T2+1,0);

    for(int i=0; i<=T1; i++) {
	for(int j=0; j<=T2; j++) {
	    Matrix<uint8_t> thresholded_img_c1 = py_binary_threshold_image_2d(image_c1, i);
	    Matrix<uint8_t> thresholded_img_c2 = py_binary_threshold_image_2d(image_c2, j);
	    Matrix<uint8_t> Kij = elementwise_AND_2d(thresholded_img_c1, thresholded_img_c2);
	    ecp(i,j) = ec_binary_image_2d_naive(Kij);
	}
    }

    return py::array_t<int>(
	{ecp.rows, ecp.cols},
	{ecp.cols*sizeof(int), sizeof(int)},
	ecp.data.data()
    );
}

std::bitset<8> neigh_matrix(Matrix<int> &image, int i, int j, int f) {
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

std::bitset<8> neigh_matrix_lex(Matrix<int> &image, int i, int j, int f) {
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

Matrix<int> pad_img_2d(py::array_t<int> image, int val) {
    auto img = image.unchecked<2>();

    size_t N = img.shape(0);
    size_t M = img.shape(1);

    Matrix<int> padded_img(N+2, M+2, val);

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

    Matrix<int> ecp(T1+1, T2+1, 0);

    Matrix<int> padded_img_c1 = pad_img_2d(image_c1, T1+1);
    Matrix<int> padded_img_c2 = pad_img_2d(image_c2, T2+1);

    for(size_t i=1; i < N+1; ++i) {
	for(size_t j=1; j<M+1; ++j) {
	    int r = padded_img_c1(i,j); 
	    std::bitset<8> binary_neigh1 = neigh_matrix_lex(padded_img_c1, i, j, r);

	    std::vector<int> thresholds2(10, T2+1);
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
        {ecp.rows, ecp.cols},       // Shape
	{ecp.cols*sizeof(int),sizeof(int)},
	ecp.data.data()
    );
}

int euler_change_neigh_matrix(Matrix<uint8_t> neigh) {
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

Matrix<int> euler_changes_2d() {
    Matrix<int> euler_changes(256,1,0);

    for(int i=0; i<256; i++) {
	euler_changes(i,0) = euler_change_neigh_matrix(i);
    }

    return euler_changes;
}

py::array_t<int> py_euler_changes_2d() {
    Matrix<int> euler_changes = euler_changes_2d();

    return py::array_t<int>(
	256,
	euler_changes.data.data()
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
}
