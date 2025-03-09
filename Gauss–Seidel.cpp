#include <iostream>
#include "tmatrix.h"
#include <string>
#include <time.h>
#include <algorithm>

/*
//Ax = b
template<typename type>
TDynamicVector<type> Gauss_Seidel_it(TDynamicMatrix<type> A, TDynamicVector<type> b, TDynamicVector<type> x, size_t num) {
	TDynamicVector<type> temp(b.size());
	temp = x;
	for (size_t n = 0; n < num; ++n) {
		for (size_t i = 0; i < b.size(); ++i) {
			x[i] = b[i];

			for (size_t j = 0; j < i; ++j) {
				x[i] -= A[i][j] * x[j];
			}

			for (size_t j = i + 1; j < b.size(); ++j) {
				x[i] -= A[i][j] * temp[j];
			}

			x[i] *= (1 / A[i][i]);
		}
		temp = x;
	}
	return x;
}
*/

template<typename type>
TDynamicVector<type> Gauss_Seidel_type(TDynamicMatrix<type> A, TDynamicVector<type> b, TDynamicVector<type> x, type ref = 1.e-8) {
	TDynamicVector<type> temp(b.size());

	//initial approximation
	temp = x;

	size_t count_it = 0;
	while(!CloseSol(A, x, b, ref)) {
		for (size_t i = 0; i < b.size(); ++i) {
			x[i] = b[i];

			for (size_t j = 0; j < i; ++j) {
				x[i] -= A[i][j] * x[j];
			}

			for (size_t j = i + 1; j < b.size(); ++j) {
				x[i] -= A[i][j] * temp[j];
			}

			x[i] *= (1 / A[i][i]);
		}
		temp = x;
		count_it++;
	}
	return x;
}

/*
TDynamicVector<double> Gauss_Seidel(TDynamicMatrix<double> A, TDynamicVector <double> b, size_t it_count, double ref = 1.e-8) {
	TDynamicMatrix<float> A_fp32(A.size());
	TDynamicVector<float> x_fp32(b.size());
	TDynamicVector<float> b_fp32(b.size());

	//initial approximation
	for (size_t i = 0; i < x_fp32.size(); ++i) x_fp32[i] = 1.f;

	A_fp32 = TDynamicMatrix<float>(A);
	b_fp32 = TDynamicVector<float>(b);
	x_fp32 = Gauss_Seidel_it(A_fp32, b_fp32,x_fp32,it_count);


	TDynamicMatrix<double> A_fp64(A.size());
	TDynamicVector<double> x_fp64(b.size());
	TDynamicVector<double> b_fp64(b.size());

	A_fp64 = TDynamicMatrix<float>(A_fp32);
	b_fp64 = TDynamicVector<float>(b_fp32);
	x_fp64 = TDynamicVector<float>(x_fp32);

	x_fp64 = Gauss_Seidel_type(A_fp64, b_fp64, x_fp64, ref);

	return x_fp64;
}*/


TDynamicVector<double> Gauss_Seidel(TDynamicMatrix<double> A, TDynamicVector <double> b, double ref1, double ref2) {
	TDynamicMatrix<float> A_fp32(A.size());
	TDynamicVector<float> x_fp32(b.size());
	TDynamicVector<float> b_fp32(b.size());

	//initial approximation
	for (size_t i = 0; i < x_fp32.size(); ++i) x_fp32[i] = 1.f;

	A_fp32 = TDynamicMatrix<float>(A);
	b_fp32 = TDynamicVector<float>(b);
	x_fp32 = Gauss_Seidel_type(A_fp32, b_fp32, x_fp32, float(ref1));


	TDynamicMatrix<double> A_fp64(A.size());
	TDynamicVector<double> x_fp64(b.size());
	TDynamicVector<double> b_fp64(b.size());

	A_fp64 = TDynamicMatrix<float>(A_fp32);
	b_fp64 = TDynamicVector<float>(b_fp32);
	x_fp64 = TDynamicVector<float>(x_fp32);

	x_fp64 = Gauss_Seidel_type(A_fp64, b_fp64, x_fp64, ref2);

	return x_fp64;
}


int main() {
	time_t t,t1,t2,t3,t4,t5;
	TDynamicMatrix<double> A(600);
	TDynamicVector<double> b(A.size());
	TDynamicVector<double> x(A.size());


	for (size_t i = 0; i < 100; ++i) {
		A.generateGoodMatrix();
		//A = generateGoodMatrix<double>(A.size());
		b.generate();


		t = clock();
		x = Gauss_Seidel(A, b, 1.e-4, 1.e-10);
		t -= clock();
		t1 = t *= -1;
		cout << double(t) <<", ";

		t = clock();
		x = Gauss_Seidel(A, b, 1.e-5, 1.e-10);
		t -= clock();
		t2=t *= -1;
		cout << double(t) << ", ";


		t = clock();
		x = Gauss_Seidel(A, b, 1.e-6, 1.e-10);
		t -= clock();
		t3=t *= -1;
		cout << double(t) << ", ";


		t = clock();
		x = Gauss_Seidel(A, b, 1.e-7, 1.e-10);
		t -= clock();
		t4=t *= -1;
		cout << double(t) << ", ";

		t = clock();
		x = Gauss_Seidel(A, b, 1.e-8, 1.e-10);
		t -= clock();
		t5=t *= -1;
		cout << double(t) << endl;

	}
	return 0;
}

//optimal for float 1.e-6