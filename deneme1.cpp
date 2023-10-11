#include <iostream>
#include <cstdint>




int main() {
    // Create an array to represent the finite field table for size 256
    uint8_t finiteField[7];

    // Populate the array with values from 0 to 255
    for (int i = 0; i < 7; i++) {
        finiteField[i] = static_cast<uint8_t>(i);
    }

    // Display the finite field table
    for (int i = 0; i < 7; i++) {
        std::cout << "Element " << i << ": " << static_cast<int>(finiteField[i]) << std::endl;
    }

    return 0;
}