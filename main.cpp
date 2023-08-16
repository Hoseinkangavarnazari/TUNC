#include <iostream>
#include "headers/ff.h"
#include <vector>
#include <stdlib.h>

int main()
{
    std::cout << "Hello World!\n";

    ff f1(256);
    uint8_t f22 = f1.add(12, 3);

    uint8_t f23 = f1.mutiplicationInverse(12);

    std::cout << "value: " << int(f22) << std::endl;

    // for (int i = 1; i < 256; i++)
    // {
    //     std::cout << "i = " << i << " = " << int(f1.division(10, uint8_t(i))) << std::endl;
    // }
    for (int i = 1; i < 256; i++)
    {
        std::cout << "i = " << i << " = " << int(f1.add(10, uint8_t(i))) << std::endl;
    }
    // std::vector<uint8_t> v1 = {1,2,3,4,5,6,7,8,9,10};
    // uint8_t a2= 12;

    // f1.s2vMultiplication(a2,v1);
}
