#include "../cFunctions.h"
#include "../packet.h"
#include "../rlnc_decoder.h"
#include <iostream>

bool rlnc_decoder::consumeCodedPacket(packet &p)
{
    this->stored_packets.push_back(p);

    return true;
}

bool rlnc_decoder::decode()
{

    // check if the rank is equal to the generation size

    this->decoder_rank = generation_size;
    if (this->decoder_rank != this->generation_size)
    {
        std::cerr << "Rank is not equal to generation size" << std::endl;
        return false;
    }

    this->stored_packets;
    this->printAugemntedMatrix();

    // how to swap two vectors 
    std::swap(this->stored_packets[0], this->stored_packets[1]);

    std::cout << std::endl
              << std::endl
              << std::endl;

    this->printAugemntedMatrix();

    // gaussin elimination

    return true;
}

void rlnc_decoder::printAugemntedMatrix()
{

    for (int pktCounter = 0; pktCounter < this->stored_packets.size(); pktCounter++)
    {
        // Print vector elements on one line[]

        std::string temp = "";
        for (int i = 0; i < this->generation_size; ++i)
        {
            temp += std::to_string(int(this->stored_packets[pktCounter].coeffcients[i])) + " ";
        }

        temp.resize(50, ' ');
        std::cout << temp << "\t|\t";

        for (int i = 0; i < this->symbol_size; ++i)
        {
            std::cout << int(this->stored_packets[pktCounter].codedSymbol[i]) << " ";
        }
        std::cout << std::endl;
    }
}