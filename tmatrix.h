// ����, �����, ���� "��������� � ��������� ������"
//
// Copyright (c) ������ �.�.
//
//

#ifndef __TDynamicMatrix_H__
#define __TDynamicMatrix_H__

#include <iostream>
#include <cassert>
#include <random>
#include <iomanip>
#include <vector>

using namespace std;

const int MAX_VECTOR_SIZE = 100000000;
const int MAX_MATRIX_SIZE = 10000;

// ������������ ������ - 
// ��������� ������ �� ������������ ������
template<typename T>
class TDynamicVector
{
protected:
    size_t sz;
    //    T* pMem;
    vector<T> pMem;
public:
    TDynamicVector(size_t size = 1) : sz(size)
    {
        if (sz == 0)
            throw out_of_range("Vector size should be greater than zero");
        if (sz > MAX_VECTOR_SIZE)
            throw out_of_range("Too large vector");
        //        pMem = new T[sz]();// {}; // � ���� T �.�. ���������� �� ���������
        pMem.resize(sz);
    }
    TDynamicVector(T* arr, size_t s) : sz(s)
    {
        if (sz == 0)
            throw out_of_range("Vector size should be greater than zero");
        if (sz > MAX_VECTOR_SIZE)
            throw out_of_range("Too large vector");
        if (arr == nullptr)
            throw invalid_argument("TDynamicVector ctor requires non-nullptr arg");
        //        pMem = new T[sz];
        pMem.resize(sz);
        std::copy(arr, arr + sz, pMem);
    }
    TDynamicVector(const TDynamicVector& v)
    {
        sz = v.sz;
        //        pMem = new T[sz];
        pMem.resize(sz);
        std::copy(v.pMem.begin(), v.pMem.end(), pMem.begin());
    }
    TDynamicVector(TDynamicVector&& v) noexcept
    {
        //        pMem = nullptr;
        swap(*this, v);
    }
    ~TDynamicVector()
    {
        //        delete[] pMem;
    }
    TDynamicVector& operator=(const TDynamicVector& v)
    {
        if (this == &v) return *this;
        //        if (sz != v.sz) {
        //            T* p = new T[v.sz];
        //            delete[] pMem;
        //            sz = v.sz;
        //            pMem = p;
        //        }
        //        std::copy(v.pMem, v.pMem + sz, pMem);
        pMem = v.pMem;
        sz = v.sz;
        return *this;
    }
    TDynamicVector& operator=(TDynamicVector&& v) noexcept
    {
        swap(*this, v);
        return *this;
    }

    size_t size() const noexcept { return sz; }

    // ����������
    T& operator[](size_t ind)
    {
        return pMem[ind];
    }
    const T& operator[](size_t ind) const
    {
        return pMem[ind];
    }
    // ���������� � ���������
    T& at(size_t ind)
    {
        if (ind >= sz) throw out_of_range("Out of range");
        return pMem[ind];
    }
    const T& at(size_t ind) const
    {
        if (ind >= sz) throw out_of_range("Out of range");
        return pMem[ind];
    }

    // ���������
    bool operator==(const TDynamicVector& v) const noexcept
    {
        if (sz != v.sz) return false;
        for (size_t i = 0; i < sz; ++i)
            if (pMem[i] != v.pMem[i])
                return false;

        return true;
    }
    bool operator!=(const TDynamicVector& v) const noexcept
    {
        return !(*this == v);
    }

    // ��������� ��������
    TDynamicVector operator+(T val)
    {
        TDynamicVector tmp(sz);
        for (size_t i = 0; i < sz; i++)
            tmp.pMem[i] = pMem[i] + val;
        return tmp;
    }
    TDynamicVector operator-(T val)
    {
        TDynamicVector tmp(sz);
        for (size_t i = 0; i < sz; i++)
            tmp.pMem[i] = pMem[i] - val;
        return tmp;
    }
    TDynamicVector operator*(T val)
    {
        TDynamicVector tmp(sz);
        for (size_t i = 0; i < sz; i++)
            tmp.pMem[i] = pMem[i] * val;
        return tmp;
    }

    // ��������� ��������
    TDynamicVector operator+(const TDynamicVector& v) const
    {
        if (sz != v.sz) throw length_error("Incompatible sizes");
        TDynamicVector tmp(sz);
        for (size_t i = 0; i < sz; i++)
            tmp.pMem[i] = pMem[i] + v.pMem[i];
        return tmp;
    }
    TDynamicVector operator-(const TDynamicVector& v) const
    {
        if (sz != v.sz) throw length_error("Incompatible sizes");
        TDynamicVector tmp(sz);
        for (size_t i = 0; i < sz; i++)
            tmp.pMem[i] = pMem[i] - v.pMem[i];
        return tmp;
    }
    T operator*(const TDynamicVector& v) const
    {
        if (sz != v.sz) throw length_error("Incompatible sizes");
        T tmp = T();
        for (size_t i = 0; i < sz; i++)
            tmp += pMem[i] * v.pMem[i];
        return tmp;
    }

    friend void swap(TDynamicVector& lhs, TDynamicVector& rhs) noexcept
    {
        std::swap(lhs.sz, rhs.sz);
        std::swap(lhs.pMem, rhs.pMem);
    }

    // ����/�����
    friend istream& operator>>(istream& istr, TDynamicVector& v)
    {
        for (size_t i = 0; i < v.sz; i++)
            istr >> v.pMem[i]; // ��������� ��������>> ��� ���� T
        return istr;
    }
    friend ostream& operator<<(ostream& ostr, const TDynamicVector& v)
    {
        for (size_t i = 0; i < v.sz; i++)
            ostr << std::setw(15) << v.pMem[i]; // ��������� ��������<< ��� ���� T
        return ostr;
    }

    double norm1() {
        double res = 0.0;
        for (size_t i = 0; i < sz; ++i)
            res += abs(double(pMem[i]));
        return res;
    }

    void generate() {
        std::random_device r;
        std::default_random_engine e(r());
        std::uniform_real_distribution<double> coef_gen(-1.0, 1.0);

        for (size_t i = 0; i < sz; ++i) {
            pMem[i] = coef_gen(e);
        }
    }

    template <typename type2>
    operator TDynamicVector<type2>() {
        TDynamicVector<type2> V(size());
        for (size_t i = 0; i < size(); ++i) {
            V[i] = type2((*this)[i]);
        }
        return V;
    }
};


// ������������ ������� - 
// ��������� ������� �� ������������ ������
template<typename T>
class TDynamicMatrix : private TDynamicVector<TDynamicVector<T>>
{
    using TDynamicVector<TDynamicVector<T>>::pMem;
    using TDynamicVector<TDynamicVector<T>>::sz;
public:
    TDynamicMatrix(size_t s = 1) : TDynamicVector<TDynamicVector<T>>(s)
    {
        if (sz > MAX_MATRIX_SIZE)
            throw out_of_range("Too large matrix");
        for (size_t i = 0; i < sz; i++)
            pMem[i] = TDynamicVector<T>(sz);
    }

    using TDynamicVector<TDynamicVector<T>>::size;
    using TDynamicVector<TDynamicVector<T>>::operator[];
    using TDynamicVector<TDynamicVector<T>>::at;

    // ���������
    bool operator==(const TDynamicMatrix& m) const noexcept
    {
        return TDynamicVector<TDynamicVector<T>>::operator==(m);
    }

    bool operator!=(const TDynamicMatrix& m) const noexcept // my func
    {
        return TDynamicVector<TDynamicVector<T>>::operator!=(m);
    }

    // ��������-��������� ��������
    TDynamicMatrix<T> operator*(const T& val)
    {
        TDynamicMatrix<T> tmp(sz);
        for (size_t i = 0; i < sz; ++i)
            tmp.pMem[i] = pMem[i] * val;
        return tmp;
    }

    // ��������-��������� ��������
    TDynamicVector<T> operator*(const TDynamicVector<T>& v)
    {
        TDynamicVector<T> tmp(sz);
        for (size_t i = 0; i < sz; ++i)
            tmp[i] = pMem[i] * v;
        return tmp;
    }

    // ��������-��������� ��������
    TDynamicMatrix operator+(const TDynamicMatrix& m)
    {
        TDynamicMatrix tmp(sz);
        for (size_t i = 0; i < sz; i++)
            tmp.pMem[i] = pMem[i] + m.pMem[i];
        return tmp;
    }

    TDynamicMatrix operator-(const TDynamicMatrix& m)
    {
        TDynamicMatrix tmp(sz);
        for (size_t i = 0; i < sz; i++)
            tmp.pMem[i] = pMem[i] - m.pMem[i];
        return tmp;
    }
    TDynamicMatrix operator*(const TDynamicMatrix& m) //square matrixes
    {
        if (sz != m.sz) throw length_error("incompatible sizes");
        TDynamicMatrix res(sz);

        for (size_t i = 0; i < sz; ++i) {
            for (size_t k = 0; k < sz; ++k) {
                for (size_t j = 0; j < sz; ++j) {
                    res[i][j] += pMem[i][k] * m[k][j];
                }
            }
        }
        return res;
    }

    double norm1() const
    {
        double res = 0;
        double sum;
        for (size_t j = 0; j < sz; ++j) {
            sum = 0;
            for (size_t i = 0; i < sz; ++i)
                sum += std::abs(double(pMem[i][j]));
            if (sum > res) res = sum;
        }
        return res;
    }

    //void LUdecomposition(TDynamicMatrix& L, TDynamicMatrix& U) const {
    //    L = TDynamicMatrix<T>(sz);
    //    U = TDynamicMatrix<T>(sz);
    //    for (size_t i = 0; i < sz; ++i) {
    //        /*
    //        for (size_t j = 0; j < sz; ++j) {
    //            U[i][j] = 0;
    //            L[i][j] = 0;
    //        }
    //        */
    //        L[i][i] = 1;
    //    }

    //    T sum;
    //    for (size_t i = 0; i < sz; ++i) {
    //        for (size_t j = 0; j < sz; ++j) {
    //            sum = 0;
    //            if (i <= j) {
    //                for (size_t k = 0; k <= i; ++k) {
    //                    sum += L[i][k] * U[k][j];
    //                }
    //                U[i][j] = pMem[i][j] - sum;
    //            }
    //            else {
    //                for (size_t k = 0; k <= j; ++k) {
    //                    sum += L[i][k] * U[k][j];
    //                }
    //                L[i][j] = (pMem[i][j] - sum) / U[j][j];
    //            }
    //        }
    //    }
    //}

    // ����/�����
    friend istream& operator>>(istream& istr, TDynamicMatrix& m)
    {
        for (size_t i = 0; i < m.sz; ++i) {
            istr >> m.pMem[i];
        }
        return istr;
    }
    friend ostream& operator<<(ostream& ostr, const TDynamicMatrix& m)
    {
        for (size_t i = 0; i < m.sz; ++i)
            ostr << m.pMem[i] << endl;
        return ostr;
    }

    //TDynamicVector<T> solver(TDynamicVector<T>& b) const {
    //    TDynamicMatrix<T> L(sz), U(sz);
    //    LUdecomposition(L, U);
    //    TDynamicVector<T> y(sz);
    //    //        cout << endl << L << endl << U << endl;
    //            // Ly = b, L - upper triangle
    //    for (size_t i = 0; i < sz; ++i) {
    //        y[i] = b[i];
    //        for (size_t j = i; j > 0; --j) {
    //            y[i] -= y[j - 1] * L[i][j - 1];
    //        }
    //    }
    //    //        cout << y << endl << L * y << endl << endl;
    //    TDynamicVector<T> x(sz);
    //    // Ux = y, U - bottom triangle
    //    for (size_t i = sz; i > 0; --i) {
    //        x[i - 1] = y[i - 1];
    //        for (size_t j = i; j < sz; ++j) {
    //            x[i - 1] -= x[j] * U[i - 1][j];
    //        }
    //        x[i - 1] = x[i - 1] / U[i - 1][i - 1];
    //    }
    //    return x;
    //}

    void generate() {
        for (size_t i = 0; i < sz; ++i) {
            for (size_t j = 0; j < sz; ++j) {
                pMem[i][j] = 0;
            }
            pMem[i][i] = 1;
        }
        //        vector<size_t> vI;
        //        vector<size_t> vJ;
        //        vector<double> vC;

        std::random_device r;
        std::default_random_engine e(r());
        std::uniform_int_distribution<size_t> count_gen(sz, sz * sz);
        std::uniform_int_distribution<size_t> index_gen(0, sz - 1);
        std::uniform_real_distribution<double> coef_gen(-1.0, 1.0);
        size_t count = count_gen(e);
        size_t indexI, indexJ;
        double coef;

        for (size_t i = 0; i < count; ++i) {
            indexI = index_gen(e);
            indexJ = index_gen(e);
            coef = coef_gen(e);
            //            vI.push_back(indexI);
            //            vJ.push_back(indexJ);
            //            vC.push_back(coef);
            //            cout << indexI << " " << indexJ << " " << coef << endl;
            pMem[indexI] = pMem[indexI] + pMem[indexJ] * T(coef);
            //            cout << *this << endl;
        }
        //        if (norm1() >= 1'000'000.0) {
        //            for (size_t i = 0; i < vI.size(); ++i)
        //                cout << vI[i] << " " << vJ[i] << " " << vC[i] << endl;
        //            throw runtime_error("ERROR");
        //        }
    }

    template <typename type2>
    operator TDynamicMatrix<type2>(){
        TDynamicMatrix<type2> M(size());
        for (size_t i = 0; i < size(); ++i) {
            for (size_t j = 0; j < size(); ++j) {
                M[i][j] = type2((*this)[i][j]);
            }
        }
        return M;
    }


    void generateGoodMatrix() {
        std::random_device r;
        std::default_random_engine e(r());
        std::uniform_real_distribution<double> coef_gen(5.0, 100.0);  // �������� ��� ������������ ���������
        std::uniform_real_distribution<double> small_gen(-4., 4.); // ����� �������� ��� �������������� ���������


        // ���������� ��������� �������������� ������� > 1
        for (size_t i = 0; i < sz; ++i) {
            pMem[i][i] = T(coef_gen(e)); // �������������� ��������� �������� (>1)
        }

        // ���������� ��������� �������������� ���������
        for (size_t i = 0; i < sz; ++i) {
            for (size_t j = 0; j < sz; ++j) {
                if (i != j) {
                    pMem[i][j] = T(small_gen(e)); // ����� �������� ��� ����������� ������������� ���������
                }
            }
        }
        
    }

};


/*
����������� ������������ ������� �����������
//���������� ������� matrix ��� row-�� ������ � col-���� �������, ��������� � newMatrix
template <typename type>
void getMatrixWithoutRowAndCol(TDynamicMatrix<type>& matrix, int row, int col, TDynamicMatrix<type>& newMatrix) {
    size_t size = matrix.size();
    int offsetRow = 0; //�������� ������� ������ � �������
    int offsetCol = 0; //�������� ������� ������� � �������
    for (int i = 0; i < size - 1; i++) {
        //���������� row-�� ������
        if (i == row) {
            offsetRow = 1; //��� ������ ��������� ������, ������� ���� ����������, ������ �������� ��� �������� �������
        }

        offsetCol = 0; //�������� �������� �������
        for (int j = 0; j < size - 1; j++) {
            //���������� col-�� �������
            if (j == col) {
                offsetCol = 1; //��������� ������ �������, ��������� ��� ���������
            }

            newMatrix[i][j] = matrix[i + offsetRow][j + offsetCol];
        }
    }
}


template <typename type>
type matrixDet(TDynamicMatrix<type> matrix) {
    type det = 0;
    int degree = 1; // (-1)^(1+j) �� ������� ������������
    size_t size = matrix.size();

    //������� ������ �� ��������
    if (size == 1) {
        return matrix[0][0];
    }
    //������� ������ �� ��������
    else if (size == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    else {
        //������� ��� ������ � �������
        TDynamicMatrix<type> newMatrix(size-1);

        //������������ �� 0-�� ������, ���� ����� �� ��������
        for (int j = 0; j < size; j++) {
            //������� �� ������� i-� ������ � j-�� �������
            //��������� � newMatrix
            getMatrixWithoutRowAndCol(matrix, 0, j, newMatrix);

            //����������� �����
            //�� �������: ����� �� j, (-1)^(1+j) * matrix[0][j] * minor_j (��� � ���� ����� �� �������)
            //��� minor_j - �������������� ����� �������� matrix[0][j]
            // (�������, ��� ����� ��� ������������ ������� ��� 0-�� ������ � j-�� �������)
            det = det + (degree * matrix[0][j] * matrixDet(newMatrix));
            //"�����������" ������� ���������
            degree = -degree;
        }

        //������ ������ �� ������ ���� ��������(�����!)
        //delete newMatrix;
    }

    return det;
}
*/
template <typename type>
type Det(TDynamicMatrix<type> matrix)
{
    type det = 0;
    int sign = 1;
    size_t size = matrix.size();
    // Base Case
    if (size == 1) {
        det = matrix[0][0];
    }
    else if (size == 2) {
        det = (matrix[0][0] * matrix[1][1])
            - (matrix[0][1] * matrix[1][0]);
    }

    // Perform the Laplace Expansion
    else {
        for (int i = 0; i < size; i++) {

            // Stores the cofactor matrix
            TDynamicMatrix<type> cofactor(size-1);
            int sub_i = 0, sub_j = 0;
            for (int j = 1; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    if (k == i) {
                        continue;
                    }
                    cofactor[sub_i][sub_j] = matrix[j][k];
                    sub_j++;
                }
                sub_i++;
                sub_j = 0;
            }

            // Update the determinant value
            det += sign * matrix[0][i] * Det(cofactor);
                sign = -sign;
        }
    }

    // Return the final determinant value
    return det;
}
template <typename type>
bool IsDiagDiffFromZero(TDynamicMatrix<type> M, type ref) {
    for (size_t i = 0; i < M.size(); ++i) {
        if (M[i][i] <= ref) return false;
    }
    return true;
}

template <typename type>
TDynamicVector<type> DiagMatrix(TDynamicMatrix<type> M) {
    TDynamicVector<type> Diag(M.size());
    for (size_t i = 0; i < M.size(); ++i) Diag[i] = M[i][i];
    return Diag;
}

template <typename type>
bool CloseSol(TDynamicMatrix<type> A, TDynamicVector<type> x, TDynamicVector<type> b, type ref) {
    for (size_t i = 0; i < A.size(); ++i) {
        if ((A * x - b)[i] >= ref) return false;
    }
    return true;
}

template <typename type>
bool IsMatrixDiffFromInf(TDynamicMatrix<type> M, type ref) {
    for (size_t i = 0; i < M.size(); ++i) {
        for (size_t j = 0; j < M.size(); ++j) {
            if (M[i][j] >= ref) return false;
        }
    }
    return true;
}




#endif
