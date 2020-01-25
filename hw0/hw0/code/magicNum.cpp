#include <iostream>
#include <cmath>
using namespace std;

int GetInput()
{
	int input;
	cout << "Enter a 3-digit number: ";
	if (!cin >> input)
	{
		cerr << "Invalid input"<< endl;
		return 0;
	}
	else
	{
		return input;
	}
}

int * changeSeq(int input){
	int seq[1];
	int last,mid,first;

	last = input % 100;
	first = ceil(input /100);
	mid = ceil(input /10);

	seq1 = (last*100) + (first*10) + (mid);
	seq2 = (mid*100) + (last*10) + (first);

	seq[0] = seq1;
	seq[1] = seq2;

	return seq;
}

int operatn(int * nums){

	int rem1,rem2,rem3;
	int sum1,sum2,sum3;
	
	rem1 = input % 11;
	rem2 = seq1 % 11;
	rem3 = seq2 % 11;
}

int main(int argc, char const *argv[])
{
	int ogNum = GetInput();
	if (ogNum == 0) {
		cerr << "Invalid input"<< endl;
		break;

	else{

		 = findrem();
	}
	}
	return 0;
}