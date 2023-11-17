#include "../hpacket.h"
#include <iostream>
#include <algorithm>
// #include "../packet.h"
// #include "../keygenerator.h"
#include <vector>
#include "../ff.h"
// #include "../mac_calculator.h"
// #include "../sign_calculator.h"

ff fff(256);


////////////////////////MAC Calculator/////////////////////////////////////////////////
void hpacket::macCalculator()
{
  // ff ff(256);

  // std::vector<uint8_t> MACs;

  for (int i = 0; i < number_of_mac; i++)
  {
    uint8_t currentMAC = 0;

    for (int j = 0; j < this->h_codedSymbol.size(); j++)
    {
      currentMAC += fff.mutiply(publickeyset[i][j], h_codedSymbol[j]);
    }
    // add negative sign
    currentMAC = fff.mutiply(fff.additionInverse(1), currentMAC);
    // do the division
    currentMAC = fff.division(currentMAC, publickeyset[i][publickeyset[i].size() - 1]);
    this->MACs.push_back(currentMAC);
  }
  uint8_t res = fff.division(fff.additionInverse(3), 2);

  // std::cout << "res";
};
//////////////////////Sign Calculator//////////////////////////////////////////////////
uint8_t hpacket::signCalculator()
{
  // ff ff(256);

  // std::vector<uint8_t> MACs;

  // for (int i = 0; i < 1; i++)
  //{
  // uint8_t currentsign = 0;

  for (int j = 0; j < number_of_mac; j++)
  {
    uint8_t akif = fff.mutiply(privateKey[j], MACs[j]);
    this->c_sign.push_back(akif);
    hosein = hosein ^ c_sign[j];
  }
  // add negative sign
  currentsign = fff.mutiply(fff.additionInverse(1), hosein);
  // do the division
  currentsign = fff.division(currentsign, privateKey[privateKey.size() - 1]);
  sign = currentsign;
  return sign;
};

////////////////////////MAC Verifier/////////////////////////////////////////////////
// std::vector<std::vector<uint8_t>> &_receivedpackets
bool hpacket::macVerifier()
{
  // ff ff(256);
  ///////////////////////////////////

  //for (int i = 0; i < number_of_mac; i++)
  //{
    uint8_t currentMAC_ver = 0;
    uint8_t crnt_mltp_mac = 0;

    for (int j = 0; j < this->h_codedSymbol.size(); j++)
    {
      currentMAC_ver += fff.mutiply(publickeyset[1][j], h_codedSymbol[j]); // working
    }
    crnt_mltp_mac = fff.mutiply(publickeyset[1][publickeyset[1].size() - 1], MACs[1]);
    this->MACs_multi_verify.push_back(crnt_mltp_mac);
    this->verifier_MACs.push_back(currentMAC_ver); // working
  //}
  this->MACs_result = fff.v2vAddition(verifier_MACs, MACs_multi_verify);

  bool MAC_resultsFlag = true;
  for (int i = 0; i < MACs_result.size(); i++)
  {
    if (this->MACs_result[0] != 0)
    {
      MAC_resultsFlag = false;
      return MAC_resultsFlag;
    }
  }
  MAC_resultsFlag = true;
  return MAC_resultsFlag;
};
//};
// uint8_t res = ff.division(ff.additionInverse(3), 2);

// std::cout<<"res";
//};
//   uint8_t hpacket::signCalculator()  1
//{ 2
// uint8_t sign = 0;
// uint8_t currentsign = 0;
// uint8_t c_sign=0;
//_privateKey= private_key;
//   ff ff(256);3
//  First, privatekey with MACs and put result in sign
// for (int k = 0; k < number_of_mac; k++)4
//{ 5
// this-> currentsign.push_back(ff.mutiply(privateKey[k], MACs[k]));

// currentsign += ff.mutiply(privateKey[k], MACs[k]) ; 6
//} 7
// currentsign = ff.mutiply(ff.additionInverse(1), currentsign);8
// currentsign = ff.division(currentsign, privateKey[privateKey.size()-1]);9

//   std::vector<uint8_t> signn = ff.v2vMulipllication(_privatekey, MAC);
// sign=currentsign; 10
// Second, divide  sign by the last entry of privatekey and update sign
// sign = ff.division(ff.additionInverse(sign),privateKey[number_of_mac]);
// return sign;11
//};12
///////////////////////////Sign Verifier///////////////////////////////////////
bool hpacket::signVerifier()
{
  // ff ff(256);

  for (int j = 0; j < number_of_mac; j++)
  {
    uint8_t akif_verifier = fff.mutiply(privateKey[j], MACs[j]);
    this->c_sign_result1.push_back(akif_verifier);
    hosein_result = hosein_result ^ akif_verifier;
  };
  //  First, multiply privatekey with received mac and put result in sign result vector
  c_sign_result2 = fff.mutiply(privateKey[privateKey.size() - 1], sign);
  sign_result = c_sign_result2 ^ hosein_result;

  bool sign_resultsFlag = true;
  // for (int i = 0; i < sign_result.size(); i++)
  //{
  if (this->sign_result != 0)
  {
    sign_resultsFlag = false;
  }
    return sign_resultsFlag;
  
  //}
  // bool MAC_resultsFlag = true;
  // return MAC_resultsFlag;
};
void hpacket::packetAppender()
{
  h_appendedSymbol = h_codedSymbol;
  h_appendedSymbol.insert(h_appendedSymbol.end(), MACs.begin(), MACs.end());
  h_appendedSymbol.insert(h_appendedSymbol.end(),sign);
};

void hpacket::packetCombiner()
{
for (int i = 0; i < h_appendedSymbol.size(); i++)
{
h_combinedSymbol[i]= fff.add(h_appendedSymbol[i],h_appendedSymbol[i]);
};
};

// Second, sum sign_result with
// sign_result = ff.add(sign_result, _privateKey.back());1
// bool allZero = true;2
// if (sign_result != 0)3
//{4
//   allZero = false;5
//}6
//  int sign_ver_result = (allZero ? 0 : 1);7
//};8

hpacket::hpacket(std::vector<uint8_t> _codedSymbol,std::vector<uint8_t> _codedSymbol2, std::vector<uint8_t> _MAC, std::vector<std::vector<uint8_t>> _publicKeySet, std::vector<uint8_t> _privateKey, int _numberofMac)
{
  this->h_codedSymbol = _codedSymbol;
    this->h_codedSymbol2 = _codedSymbol2;

  this->privateKey = _privateKey;
  this->publickeyset = _publicKeySet;
  this->number_of_mac = _numberofMac;
  this->MACs = _MAC;
  // check if the coded symbol is not empty
  // check the size of publickeyset mateches to the number_of_macs
  // call the mac calculator to calculate the

  this->macCalculator();
  this->signCalculator();
  // this->macVerifier();
  // this->signVerifier();
};
