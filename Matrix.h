#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <iomanip>

using namespace std;

class Matrix{
private:
	int rows;
	int col;

public:
	vector<vector<double>> data;

	Matrix();
	Matrix(int r, int c);
	Matrix(vector<vector<double>> d);
	Matrix(int r, int c, bool isRandom);

	int getRows() const {return rows;}
	int getCol() const {return col;}
	vector<vector<double>> getData() const {return data;}

	void print_matrix() const;

	Matrix operator*(const Matrix& mat) const;
	Matrix operator+(const Matrix& mat) const;
	Matrix operator-(const Matrix& mat) const;
	Matrix operator*(double num) const;
	Matrix exp_multiply()const;
	Matrix scalar_subtraction(double num) const;

	Matrix transpose() const;
	Matrix element_multiply(const Matrix& mat) const;
	Matrix element_divide(const Matrix& mat) const;
	Matrix element_sqrt() const;
	Matrix operator/(double num) const;
};

#endif 