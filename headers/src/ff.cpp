#include "../ff.h"

// bool ff::generateTable(){
//     cout<<"generateTable works!"<<endl;
//     return true;
// }

// std::vector<uint8_t> ff::s2vMultiplication(uint8_t a, std::vector<uint8_t> b)
// {
//     return b;
// }


uint8_t ff::mutiply(uint8_t a, uint8_t b)
{

    return ff256[a][b];
}