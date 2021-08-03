#include <cmath>
#include <iostream>
#include <complex>
#include <vector>

const double pi = acos(-1.0);

std::vector<std::complex<double>> fft(std::vector<std::complex<double>>& input) {
    std::vector<std::complex<double>> output;
    for (size_t k = 0; k < input.size(); k++) {
        std::complex<double> sum(0, 0);
        for (size_t n = 0; n < input.size(); n++) {
            std::complex<double> wn(std::polar<double>(1, -2 * pi / N * n * k));
            sum += input[n] * wn;
        }
        output.push_back(sum);
    }
    return output;
}


void FFT_recursion(std::vector<std::complex<double>>& input) 
{
    size_t n = input.size();
    if(n == 1)
     return;
    size_t mid = input.size() / 2;
    std::vector<std::complex<double>> A0, A1;
    for(int i = 0; i < n; i += 2) {//拆分奇偶下标项
        A0.push_back(input[i]);
        A1.push_back(input[i + 1]);
    }
    FFT_recursion(A0);
    FFT_recursion(A1);
    std::complex<double> w0(1, 0);
    std::complex<double> wn(std::polar<double>(1, -2 * pi / n));//单位根
    for(int i=0; i < mid; i++, w0 *= wn) {//合并多项式
        input[i] = A0[i] + w0*A1[i];
        input[i + mid] = A0[i] - w0*A1[i];
    }
}



 void  dit2(std::complex<double>* Data, int  Log2N, int  sign)
{
    int i,j,k,step,length;
    std::complex<double> wn,temp,deltawn;
    length=1<<Log2N;
    for(i=0;i<length;i+=2)
    {
        temp=Data[i];a
        Data[i]=Data[i]+Data[i+1];
        Data[i+1]=temp-Data[i+1];
    }
    for(i=2;i<=Log2N;i++)
    {
        wn=1;step=1<<i;deltawn=std::complex<double>(cos(2.0*pi/step),sin(sign*2.0*pi/step));；
        for(j=0;j<step/2;j++)
        {        
        for(i=0;i<length/step;i++)
        {
            temp=Data[i*step+step/2+j]*wn;
            Data[i*step+step/2+j]=Data[i*step+j]-temp;
            Data[i*step+j]=Data[i*step+j]+temp;
            }
            wn=wn*deltawn;
        }
    }
    if(sign==1)
    for(i=0;i<length;i++)
    Data[i]/=length;
}




int main()
{
    std::vector<std::complex<double>> in{1, 2, 3};
    auto re = fft(in);
    for (auto i: re)
    {
        std::cout << "real:" << i.real() << " imag:" << i.imag() << std::endl;
    }
    return 0;
}
