#include "../hpacket.h"
#include <iostream>
#include <algorithm>
// #include "../packet.h"
// #include "../keygenerator.h"
#include <vector>
#include "../ff.h"
#include <cmath>
#include <cstdlib> // Include the necessary header for rand() and srand()
#include <ctime>   // Include the necessary header for time()
// #include "../mac_calculator.h"
// #include "../sign_calculator.h"
#include <numeric>


ff fff(256);

uint8_t hpacket::powerCalculator(uint8_t k , uint8_t n)
{
  uint8_t powerResult=1;
  if (n>0)
    {
  for(int j=0; j<n ; j++)
  {
    
    powerResult = fff.mutiply(k,powerResult);
  };};
  return powerResult;
};

void hpacket::multiplyCheck()
{
   uint8_t cnt=0;
   for (int j=0; j<256; j++)
   {
    this->div_result.push_back(fff.division(powerCalculator(generator,cnt),powerCalculator(generator,cnt))); 
    this->mult_result.push_back(fff.mutiply(powerCalculator(generator,cnt),powerCalculator(generator,cnt)));
    this->mult_inv_result.push_back(fff.mutiply(powerCalculator(generator,cnt),fff.mutiplicationInverse(powerCalculator(generator,cnt))));
    this->add_inv_table.push_back(fff.additionInverse(cnt)); 
    cnt=cnt+1;
   };
};

////////////////////////MAC Calculator/////////////////////////////////////////////////
void hpacket::macCalculator()
{
  // ff ff(256);

  // std::vector<uint8_t> MACs;
for(int pcktIndex=0; pcktIndex<h_codedSymbol.size(); pcktIndex++){

    std::vector<uint8_t> MACvector(number_of_mac);

  for (int i = 0; i < number_of_mac; i++)
  {
    uint8_t currentMAC = 0;

  
    for (int j = 0; j < h_codedSymbol[0].size(); j++)
    {
      uint8_t MAC_mult_sum = fff.mutiply(publickeyset[i][j], h_codedSymbol[pcktIndex][j]);

      currentMAC = currentMAC^MAC_mult_sum;

    };
    // add negative sign

    currentMAC = fff.mutiply(fff.additionInverse(1), currentMAC);

    // do the division
    currentMAC = fff.division(currentMAC, publickeyset[i][publickeyset[i].size()-1]);
                  //std::cout << "here";
    MACvector[i]= currentMAC;

  };
  this->MACs.push_back(MACvector);

 // uint8_t res = fff.division(fff.additionInverse(3), 2);

  // std::cout << "res";
};};
//////////////// MAC Calculator for one packet
std::vector<uint8_t> hpacket::macCalculatorONEPACKET(std::vector<uint8_t> _current_packet, std::vector<std::vector<uint8_t>> _keypool)
{
  // ff ff(256);

  // std::vector<uint8_t> MACs;


    std::vector<uint8_t> MACvector(number_of_mac);

  for (int i = 0; i < number_of_mac; i++)
  {
    uint8_t currentMAC = 0;

  
    for (int j = 0; j < _current_packet.size(); j++)
    {
      uint8_t MAC_mult_sum = fff.mutiply(_keypool[i][j], _current_packet[j]);

      currentMAC = currentMAC^MAC_mult_sum;

    };
    // add negative sign

    currentMAC = fff.mutiply(fff.additionInverse(1), currentMAC);

    // do the division
    currentMAC = fff.division(currentMAC, _keypool[i][_keypool[i].size()-1]);
                  //std::cout << "here";
    MACvector[i]= currentMAC;

  };

 // uint8_t res = fff.division(fff.additionInverse(3), 2);
  return MACvector;
  // std::cout << "res";
};
//////////////////////Sign Calculator//////////////////////////////////////////////////
void hpacket::signCalculator()
{

  std::vector<uint8_t> sgn(h_codedSymbol.size());
for(int pcktIndex=0; pcktIndex<h_codedSymbol.size(); pcktIndex++){
  for (int t = 0; t < number_of_mac; t++)
  {
    uint8_t sign_multiply = fff.mutiply(privateKey[t], MACs[pcktIndex][t]);
   // this->c_sign.push_back(sign_multiply);
    sign_sum = sign_sum ^ sign_multiply;
  };
  // add negative sign
  currentsign = fff.mutiply(fff.additionInverse(1), sign_sum);
  // do the division
  sgn[pcktIndex] = fff.division(currentsign, privateKey[ privateKey.size()-1 ] );
  //sign = currentsign;
};
sign = sgn;
};
///////// Sign Calculator for one packet
uint8_t hpacket::signCalculatorONEPACKET(std::vector<uint8_t> _current_packet_MACs, std::vector<uint8_t> _private_key)
{

uint8_t sgn=0;
uint8_t sign_sum=0;

  for (int t = 0; t < _private_key.size()-1; t++)
  {
    uint8_t sign_multiply = fff.mutiply(_private_key[t], _current_packet_MACs[t]);
   // this->c_sign.push_back(sign_multiply);
    sign_sum = sign_sum ^ sign_multiply;
  };
  // add negative sign
  int currentsign = fff.mutiply(fff.additionInverse(1), sign_sum);
  // do the division
  sgn = fff.division(currentsign, _private_key[ _private_key.size()-1 ] );
  //sign = currentsign;
return sgn;
};
//////////////////////////////////////
std::vector<std::vector<uint8_t>> hpacket::packetAppender(std::vector<std::vector<uint8_t>> _h_appendedSymbol )
{
  _h_appendedSymbol = h_codedSymbol;
  for(int pcktIndex=0; pcktIndex< h_codedSymbol.size() ; pcktIndex++){
  _h_appendedSymbol[pcktIndex].insert(_h_appendedSymbol[pcktIndex].end(), MACs[pcktIndex].begin(), MACs[pcktIndex].end());
  _h_appendedSymbol[pcktIndex].insert(_h_appendedSymbol[pcktIndex].end(),sign[pcktIndex]);

  };
    return _h_appendedSymbol;

};
///////////////////// Packet appender one packet/////////////////////////////////
std::vector<uint8_t> hpacket::packetAppenderONEPACKET(std::vector<uint8_t> _h_appendedPacket, std::vector<uint8_t> _macVector, uint8_t _sign )
{
  _h_appendedPacket.insert(_h_appendedPacket.end(), _macVector.begin(), _macVector.end());
  _h_appendedPacket.insert(_h_appendedPacket.end(), _sign);

    return _h_appendedPacket;

};
////////////////////////MAC Verifier/////////////////////////////////////////////////
// std::vector<std::vector<uint8_t>> &_receivedpackets
bool hpacket::macVerifier(std::vector<uint8_t> verifiedDataPacket,std::vector<std::vector<uint8_t>> _assignedKeyset)
{
  // ff ff(256);
  ///////////////////////////////////
   std::vector<uint8_t> zeroVector(_assignedKeyset.size(), 0);
   MACs_multi_verify= zeroVector;
   verifier_MACs= zeroVector;

  for (int i = 0; i < _assignedKeyset.size(); i++)
  {
    uint8_t currentMAC_ver = 0;
    uint8_t crnt_mltp_mac = 0;
    
    for (int j = 0; j <  _assignedKeyset[0].size()-1; j++)
    {

      currentMAC_ver = currentMAC_ver ^ fff.mutiply(_assignedKeyset[i][j], verifiedDataPacket[j]); // working
       //   std::cout << "here";

    };
    crnt_mltp_mac = fff.mutiply(_assignedKeyset[i][_assignedKeyset[i].size() - 1], verifiedDataPacket[_assignedKeyset[0].size()+i-1]); 
    MACs_multi_verify[i]=crnt_mltp_mac;
    verifier_MACs[i]=currentMAC_ver; // working
  };
       //     std::cout << "here";
  
  this->MACs_result = fff.v2vAddition(verifier_MACs, MACs_multi_verify);
  //  std::cout << "here";

  bool MAC_resultsFlag = true;
  for (int i = 0; i < MACs_result.size(); i++)
  {
    if (this->MACs_result[i] != 0)
    {
      MAC_resultsFlag = false;
      //return MAC_resultsFlag;
    };
  };
   // std::cout << "here";

  //MAC_resultsFlag = true;
  verification_resultsFlag = MAC_resultsFlag;
  return verification_resultsFlag;
};

bool hpacket::signVerifier(std::vector<uint8_t> verifiedDataPacket,std::vector<uint8_t> _publicKey)
{
  // ff ff(256);
//////////////////////HMAC SIGN VERIFICATION START///////////////
 // for (int j = 0; j < number_of_mac; j++)
  //{
    //multiple_sum =  powerCalculator(powerCalculator(generator,privateKey[j]),h_appendedSymbol[h_codedSymbol.size()+j]);
    //multiple_sum =  powerCalculator(fff.mutiply(privateKey[j],h_appendedSymbol[h_codedSymbol.size()+j]));
    //std::cout << "here";
    //signMultiply = fff.mutiply(multiple_sum,signMultiply );

  //};
  //uint8_t test = powerCalculator(powerCalculator(generator,privateKey[number_of_mac]),sign);
  //std::cout << "here";
  //sign_result= fff.mutiply(signMultiply,test);
    //////////////////////HMAC SIGN VERIFICATION END///////////////

  for (int j = 0; j < number_of_mac; j++)
  {
    uint8_t akif_verifier = fff.mutiply(_publicKey[j], verifiedDataPacket[verifiedDataPacket.size()-number_of_mac-1+j]);
    this->c_sign_result1.push_back(akif_verifier);
    mac_verifier_sum = mac_verifier_sum ^ akif_verifier;
  };
  //  First, multiply privatekey with received mac and put result in sign result vector
  c_sign_result2 = fff.mutiply(_publicKey[number_of_mac], verifiedDataPacket[verifiedDataPacket.size()-1]);
  sign_result = c_sign_result2 ^ mac_verifier_sum;
  // std::cout << "here";

  bool sign_resultsFlag = true;
  // for (int i = 0; i < sign_result.size(); i++)
  //{
  if (this->sign_result != 0)
  {
    sign_resultsFlag = false;
    
  };
    verification_resultsFlag = sign_resultsFlag;
    if (verification_resultsFlag = true)
    {
      this->verified_symbols.push_back(verifiedDataPacket);
    }
      //std::cout << "here";


    return verification_resultsFlag;
  
  //}
  // bool MAC_resultsFlag = true;
  // return MAC_resultsFlag;
};


////////////
uint8_t treePowerCalculator(uint8_t k , uint8_t n)
{
  uint8_t powerResult=1;
  if (n>0)
    {
  for(int j=0; j<n ; j++)
  {
    
    powerResult = k*powerResult;
  };};
  return powerResult;
};

               /////////////Pollution Generator////////////////////

std::vector<std::vector<uint8_t>> hpacket::pollutionGeneration(std::vector<std::vector<uint8_t>> received_packets_list, std::vector<int> _pollutedPacketIndex){
   
   //////Randomly select the index of the packet for pollution////////////////
       std::srand(static_cast<unsigned int>(std::time(0)));
   // received_packets_list=h_appendedSymbol;
   vectorSize=received_packets_list[0].size() ;
   //   std::cout << "here";
for(int i=0; i<_pollutedPacketIndex.size(); i++){
    int modifiedEntryIndex = std::rand() % h_codedSymbol[0].size() ;  //Data pollution
       //   std::cout << "here";
    std::vector<uint8_t> zeroVector(vectorSize, 0);      
//    std::cout << "here";
    zeroVector[modifiedEntryIndex]=1;
        //  std::cout << "here";
    received_packets_list[_pollutedPacketIndex[i]]= fff.v2vAddition(received_packets_list[_pollutedPacketIndex[i]],zeroVector);

    };
         // std::cout << "here";
    return received_packets_list;
};

std::vector<uint8_t> hpacket::pollutionGenerationONEPACKET(std::vector<uint8_t> _received_packet, int a){
   
   //////Randomly select the index of the packet for pollution////////////////
      // std::srand(static_cast<unsigned int>(std::time(0)));
   // received_packets_list=h_appendedSymbol;
   vectorSize=_received_packet.size() ;
   //   std::cout << "here";
    int modifiedEntryIndex = std::rand() % h_codedSymbol[0].size() ;  //Data pollution
       //   std::cout << "here";
    std::vector<uint8_t> zeroVector(vectorSize, a+1);      
//    std::cout << "here";
    zeroVector[modifiedEntryIndex]=modifiedEntryIndex;
        //  std::cout << "here";
    _received_packet= fff.v2vAddition(_received_packet,zeroVector);

         // std::cout << "here";
    return zeroVector;};
    //std::vector<uint8_t> hpacket:: intelligentpollutionGeneration(std::vector<uint8_t> _packet,std::vector<std::vector<uint8_t>> _assignedKeys){
      //std::vector<uint8_t> b;
      //b[0]= fff.mutiply(_assignedKeys[0][2],_packet.back());
      //b[1]= fff.mutiply(_assignedKeys[1][2],_packet.back());
     // _packet[0] = fff.add(fff.mutiply(_assignedKeys[1][1],b[0]),fff.mutiply(_assignedKeys[0][1],b[1]));

    //return 
    //};

//treeGenerator(){
//};

               /////////////Tree Generator////////////////////

std::vector<std::vector<std::vector<uint8_t>>> hpacket::treeGenerator(std::vector<std::vector<uint8_t>> received_packets_list, int _numberOfLayers,int leaves, int packet_size){

    std::vector<uint8_t> zeroVector(received_packets_list[0].size(), 0);
       std::vector<std::vector<std::vector<uint8_t>>> zero3dmatrice(_numberOfLayers, std::vector<std::vector<uint8_t>>(received_packets_list.size(), std::vector<uint8_t>(received_packets_list[0].size(), 0)));
    // _numberOfLayers = number_of_layers;
  std::vector<std::vector<std::vector<uint8_t>>>  generatedTree=zero3dmatrice;

      generatedTree[0]=received_packets_list;   //Bottom layer is the received packets

      for(int i=1; i< _numberOfLayers; i++){
         for(int j=0; j<treePowerCalculator(leaves,_numberOfLayers-i-1); j++){
            for(int k=0;k<leaves; k++){
                //if(i=0)
               // {
                 // generatedTree[i][j]= fff.v2vAddition(received_packets_list[number_of_leaves*j+k],generatedTree[i][j]);
               // };
                 generatedTree[i][j]= fff.v2vAddition(generatedTree[i-1][leaves*j+k],generatedTree[i][j]);
         
         };
         };
      };
     // std::cout << "here";
      return generatedTree;
};
               /////////// Tree Based Algorithm 2 ///////////


int hpacket::arTreeVerifier(std::vector<std::vector<std::vector<uint8_t>>> received_packets_tree, std::vector<int> ARvector, int _layer,std::vector<std::vector<uint8_t>> _assignedKeyset,std::vector<uint8_t> _publicKey){
    int ar_tree_verification_counter=1;
    bool sum_result_ar_alg=true;
    bool base_result_ar_alg=true;
    sum_result_ar_alg = macVerifier(received_packets_tree[_layer-1][0],_assignedKeyset) && signVerifier(received_packets_tree[_layer-1][0],_publicKey);     

    std::vector<int> b = ARvector; // Copy vector a to b
     // std::cout << "here";

    while(sum_result_ar_alg==false && ARvector.size()>0){
    // Sort vector a in descending order to get vector b
     std::sort(b.rbegin(), b.rend()); // Sort in descending order  
    // Find the index of the largest element in b in the original vector a
    int largestElement = b.front();
    int largestARelement= static_cast<int>(largestElement);
    int largestARindex = 0;
    for(int i=0; i<ARvector.size(); i++){
       if(ARvector[i]== largestElement){
         largestARindex = i;
         ARvector[i]=0;
       
         break;
       };
    };
      //std::cout << "here";

     base_result_ar_alg = macVerifier(received_packets_tree[0][largestARindex],_assignedKeyset) && signVerifier(received_packets_tree[0][largestARindex],_publicKey);  //check the root with highest AR
     ar_tree_verification_counter++;
      //   std::cout << "here";

     if(base_result_ar_alg==false){
      // discard the polluted root from the summation at the top layer
       received_packets_tree[_layer-1][0] = fff.v2vSubtraction(received_packets_tree[_layer-1][0],received_packets_tree[0][largestARindex]);
       sum_result_ar_alg = macVerifier(received_packets_tree[_layer-1][0],_assignedKeyset) && signVerifier(received_packets_tree[_layer-1][0],_publicKey);   
       ar_tree_verification_counter++;
      };
       //    std::cout << "here";

     b.front()=0;     

    };
             //  std::cout << "here";

  return ar_tree_verification_counter;
};

std::vector<int> hpacket::arTreeVerifierNEW(std::vector<std::vector<std::vector<uint8_t>>> received_packets_tree, std::vector<int> ARvector, int _layer,std::vector<std::vector<uint8_t>> _assignedKeyset,std::vector<uint8_t> _publicKey){
   
   // std::cout << "here";
    
    std::vector<int> arTreeVerifierNEW_output= {0,0};
    std::vector<int> updated_ar_vector = ARvector;
    int ar_tree_verification_counter=1;
    int polluted_packet_counter=0;
    bool sum_result_ar_alg=true;
    bool base_result_ar_alg=true;
    sum_result_ar_alg = macVerifier(received_packets_tree[_layer-1][0],_assignedKeyset) && signVerifier(received_packets_tree[_layer-1][0],_publicKey);     

    std::vector<int> b = updated_ar_vector; // Copy updated AR vector  to b
     // std::cout << "here";

    while(sum_result_ar_alg==false && updated_ar_vector.size()>0){
    // Sort vector a in descending order to get vector b
     std::sort(b.rbegin(), b.rend()); // Sort in descending order  
    // Find the index of the largest element in b in the original vector a
    int largestElement = b.front();
    int largestARelement= static_cast<int>(largestElement);
    int largestARindex = 0;
    for(int i=0; i<updated_ar_vector.size(); i++){
       if(updated_ar_vector[i]== largestElement){
         largestARindex = i;
         updated_ar_vector[i]=0;
         break;
       };
    };
      //std::cout << "here";

     base_result_ar_alg = macVerifier(received_packets_tree[0][largestARindex],_assignedKeyset) && signVerifier(received_packets_tree[0][largestARindex],_publicKey);  //check the root with highest AR
     ar_tree_verification_counter++;
      //   std::cout << "here";

     if(base_result_ar_alg==false){
      // discard the polluted root from the summation at the top layer
       received_packets_tree[_layer-1][0] = fff.v2vSubtraction(received_packets_tree[_layer-1][0],received_packets_tree[0][largestARindex]);
       sum_result_ar_alg = macVerifier(received_packets_tree[_layer-1][0],_assignedKeyset) && signVerifier(received_packets_tree[_layer-1][0],_publicKey);
       //ARvector[largestARindex]++;
       //path_index_vector.push_back(largestARindex);   
       ar_tree_verification_counter++;
       polluted_packet_counter++;
      };
       //    std::cout << "here";

     b.front()=0;     

    };
             //  std::cout << "here";
    arTreeVerifierNEW_output[0]=ar_tree_verification_counter;
    arTreeVerifierNEW_output[1]=polluted_packet_counter;

  return arTreeVerifierNEW_output;
};







std::vector<uint8_t> hpacket::randomCombiner(std::vector<uint8_t> _vec1,std::vector<uint8_t> _vec2){

std::vector<uint8_t> summation = fff.v2vAddition(_vec1,_vec2);

return summation;
};


    std::vector<uint8_t> hpacket::randomMultiplier(uint8_t a,std::vector<uint8_t> _vec){


        std::vector<uint8_t> mutiplication = fff.s2vMultiplication(_vec,a);

        return mutiplication;
    };

               /////////// Tree Based Algorithm 1 ///////////
int hpacket::treeVerifier(std::vector<std::vector<std::vector<uint8_t>>> received_packets_tree,int _layer,int _leaves,std::vector<std::vector<uint8_t>> _assignedKeyset,std::vector<uint8_t> _publicKey){
            int tree_verification_counter=1;

    //std::vector<uint8_t> zeroVector(number_of_mac, 0);
    //  std::cout << "here";
    std::vector<std::vector<bool>> result_vector(_layer, std::vector<bool>(treePowerCalculator(_leaves,_layer-1) , true));
          
    result_vector[0][0] = macVerifier(received_packets_tree[_layer-1][0],_assignedKeyset) && signVerifier(received_packets_tree[_layer-1][0],_publicKey);
    if(result_vector[0][0]==false){

       for(int k=0; k<_leaves; k++){
             result_vector[1][k]= false;
             };
    for(int i=1; i<_layer; i++){
        for(int j=0; j<treePowerCalculator(_leaves,i) ; j++){
          
          if(result_vector[i][j]==false){
          //  for (int k=0; k<number_of_leaves; k++){
          result_vector[i][j] = macVerifier(received_packets_tree[_layer-i-1][j],_assignedKeyset) && signVerifier(received_packets_tree[_layer-i-1][j],_publicKey);
          tree_verification_counter++;
          };
          //};
          if(result_vector[i][j]==false && i<(_layer-1)){
             for(int k=0; k<_leaves; k++){
             result_vector[i+1][_leaves*j+k]= false;
             
             };
          };
          if(result_vector[i][j]==true && i<(_layer-1)){
              for(int k=0; k<_leaves; k++){
                 result_vector[i+1][_leaves*j+k]= true;
              };
             };
             if(result_vector[i][j]==false && i==(_layer-1)){

             received_packets_tree[_layer-1][0] = fff.v2vSubtraction(received_packets_tree[_layer-1][0],received_packets_tree[0][j]);

             };
          };
        };
       result_vector[0][0] = macVerifier(received_packets_tree[_layer-1][0],_assignedKeyset) && signVerifier(received_packets_tree[_layer-1][0],_publicKey);   
      tree_verification_counter++;
     // std::cout << "here";


    };
    return tree_verification_counter;
    };

std::vector<int> hpacket::treeVerifierNEW(std::vector<std::vector<std::vector<uint8_t>>> received_packets_tree,int _layer,int _leaves,std::vector<std::vector<uint8_t>> _assignedKeyset,std::vector<uint8_t> _publicKey){
            int tree_verification_counter=1;
            int polluted_packet_counter=0;
    //std::vector<uint8_t> zeroVector(number_of_mac, 0);
    //  std::cout << "here";
    std::vector<std::vector<bool>> result_vector(_layer, std::vector<bool>(treePowerCalculator(_leaves,_layer-1) , true));
    std::vector<int> treeVerifierNEW_output = {0,0};
          
    result_vector[0][0] = macVerifier(received_packets_tree[_layer-1][0],_assignedKeyset) && signVerifier(received_packets_tree[_layer-1][0],_publicKey);
    if(result_vector[0][0]==false){

       for(int k=0; k<_leaves; k++){
             result_vector[1][k]= false;
             };
    for(int i=1; i<_layer; i++){
        for(int j=0; j<treePowerCalculator(_leaves,i) ; j++){
          
          if(result_vector[i][j]==false){
          //  for (int k=0; k<number_of_leaves; k++){
          result_vector[i][j] = macVerifier(received_packets_tree[_layer-i-1][j],_assignedKeyset) && signVerifier(received_packets_tree[_layer-i-1][j],_publicKey);
          tree_verification_counter++;
          };
          //};
          if(result_vector[i][j]==false && i<(_layer-1)){
             for(int k=0; k<_leaves; k++){
             result_vector[i+1][_leaves*j+k]= false;
             
             };
          };
          if(result_vector[i][j]==true && i<(_layer-1)){
              for(int k=0; k<_leaves; k++){
                 result_vector[i+1][_leaves*j+k]= true;
              };
             };
             if(result_vector[i][j]==false && i==(_layer-1)){

             received_packets_tree[_layer-1][0] = fff.v2vSubtraction(received_packets_tree[_layer-1][0],received_packets_tree[0][j]);
             polluted_packet_counter++;
             };
          };
        };
       result_vector[0][0] = macVerifier(received_packets_tree[_layer-1][0],_assignedKeyset) && signVerifier(received_packets_tree[_layer-1][0],_publicKey);   
      tree_verification_counter++;
     // std::cout << "here";
     treeVerifierNEW_output[0]= tree_verification_counter;
     treeVerifierNEW_output[1]=polluted_packet_counter ;

    };
    return treeVerifierNEW_output;
    };


               ///////////  Simple Algorithm      ///////////

int hpacket::simpleVerifier(std::vector<std::vector<uint8_t>> received_packets_list,std::vector<std::vector<uint8_t>> _assignedKeyset,std::vector<uint8_t> _publicKey){
 // h_appendedSymbol = received_packets_list;
  int verification_counter=0;
  std::vector<bool> simpleVerifierResult(received_packets_list.size(),true);
  for(int i=0; i<received_packets_list.size(); i++){
     bool test =false;
     bool mac_ver_result = macVerifier(received_packets_list[i],_assignedKeyset);
        //  std::cout << "here";
     bool sign_ver_result = signVerifier(received_packets_list[i],_publicKey);
              // std::cout << "here";
     if(mac_ver_result==true && sign_ver_result==true){
     test = true;
     };
     //simpleVerifierResult[i] = macVerifier(h_appendedSymbol[i]) && signVerifier(h_appendedSymbol[i]);
    simpleVerifierResult[i]=test;
          //  std::cout << "here";
     verification_counter++;
  };
  
   //     std::cout << "here";
 return verification_counter;
};

std::vector<int> hpacket::simpleVerifierNEW(std::vector<std::vector<uint8_t>> received_packets_list,std::vector<std::vector<uint8_t>> _assignedKeyset,std::vector<uint8_t> _publicKey){
 // h_appendedSymbol = received_packets_list;
  std::vector<int> simple_verifier_output={0,0};
  int verification_counter=0;
  int no_of_polluted_packets=0;
  std::vector<bool> simpleVerifierResult(received_packets_list.size(),true);
  for(int i=0; i<received_packets_list.size(); i++){
     bool test =false;
     bool mac_ver_result = macVerifier(received_packets_list[i],_assignedKeyset);
        //  std::cout << "here";
     bool sign_ver_result = signVerifier(received_packets_list[i],_publicKey);
              // std::cout << "here";
     if(mac_ver_result==true && sign_ver_result==true){
     test = true;
     };
     if((mac_ver_result==false) || (sign_ver_result==false)){
     no_of_polluted_packets++;
     };
     //simpleVerifierResult[i] = macVerifier(h_appendedSymbol[i]) && signVerifier(h_appendedSymbol[i]);
    simpleVerifierResult[i]=test;
          //  std::cout << "here";
     verification_counter++;
  };
  
   //     std::cout << "here";
   simple_verifier_output[0]= verification_counter;
   simple_verifier_output[1]= no_of_polluted_packets;
 return simple_verifier_output;
};


void hpacket::packetCombiner()
{
combination_counter=0;
for (int i = 0; i < verified_symbols.size(); i++)
{
coefficient= coefficientVector[combination_counter];
h_combinedSymbol = fff.v2vAddition(h_combinedSymbol,fff.s2vMultiplication(verified_symbols[i],coefficient));
combination_counter++;
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

hpacket::hpacket(std::vector<std::vector<uint8_t>> _codedSymbol, std::vector<std::vector<uint8_t>> _MAC, std::vector<std::vector<uint8_t>> _publicKeySet, std::vector<uint8_t> _privateKey, int _numberofMac,std::vector<uint8_t> _coefficientvector)
{
  this->h_codedSymbol = _codedSymbol;
  this->coefficientVector = _coefficientvector;
  this->privateKey = _privateKey;
  this->publickeyset = _publicKeySet;
  this->number_of_mac = _numberofMac;
  this->MACs = _MAC;
  // check if the coded symbol is not empty
  // check the size of publickeyset mateches to the number_of_macs
  // call the mac calculator to calculate the

  //this->macCalculator();
  //this->signCalculator();
  // this->macVerifier();
  // this->signVerifier();
};



/////////////////////////////////////////////////////////////////////////////////////
////////////////////////MAC Verifier/////////////////////////////////////////////////
// std::vector<std::vector<uint8_t>> &_receivedpackets

///////////////////////////////////////

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


//bool hpacket::Verifier()
//{
                                                     //for (int i = 0; i < number_of_mac; i++)
                                                     //{
  //  uint8_t current_MAC_ver = 0;
  //  uint8_t current_multiply_mac = 0;
                                                      //    std::cout << "here";


   // for (int j = 0; j < this->h_codedSymbol.size(); j++)
  //  {
  //    current_MAC_ver += fff.mutiply(publickeyset[1][j], h_appendedSymbol[j]); // working
 //   };
  //  current_multiply_mac = fff.mutiply(publickeyset[1][publickeyset[1].size() - 1], MACs[1]);
  //  this->MACs_multi_verify.push_back(current_multiply_mac);
  // this->verifier_MACs.push_back(current_MAC_ver); // working
                                                                        //}
  //this->MACs_result = fff.v2vAddition(verifier_MACs, MACs_multi_verify);
 // bool MAC_resultsFlag = true;
  //for (int i = 0; i < MACs_result.size(); i++)
 // {
  //  if (this->MACs_result[0] != 0)
  //  {
  //    MAC_resultsFlag = false;
   //   return MAC_resultsFlag;
 //   }
 // }
 // verification_resultsFlag = MAC_resultsFlag; endddddddd

                                                                                //return MAC_resultsFlag;
                                                                                 //};
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
//bool hpacket::signVerifier()
//{
  // ff ff(256);
// Our sign verifier start
  //for (int j = 0; j < number_of_mac; j++)
  //{
    //uint8_t mac_verifier_multiply = fff.mutiply(privateKey[j], h_appendedSymbol[h_codedSymbol.size()+j]);
                                                                //this->c_sign_result1.push_back(mac_verifier_multiply);
    //mac_verifier_sum = mac_verifier_sum ^ mac_verifier_multiply;
// Our sign verifier end

    // New Sign verifier from HMAC paper start
    

   
  //for (int j = 0; j < number_of_mac; j++)
 // {
 //   uint8_t akif_verifier = fff.mutiply(privateKey[j], MACs[j]);
  //  this->c_sign_result1.push_back(akif_verifier);
  //  mac_verifier_sum = mac_verifier_sum ^ akif_verifier;
 // };
  //  First, multiply privatekey with received mac and put result in sign result vector
 // c_sign_result2 = fff.mutiply(privateKey[privateKey.size() - 1], sign);
 // sign_result = c_sign_result2 ^ mac_verifier_sum;
  // New Sign verifier from HMAC paper end
                                                        //}; // end for the for loop of our sign verifier
  //  First, multiply privatekey with received mac and put result in sign result vector
  
                                                    //c_sign_result2 = fff.mutiply(privateKey[privateKey.size() - 1], sign); //our sign verifier
                                                    //sign_result = c_sign_result2 ^ mac_verifier_sum;   //our sign verifier


  //bool sign_resultsFlag = true;
                                                    // for (int i = 0; i < sign_result.size(); i++)
                                                        //{
    
  //if (this->sign_result != 0)  //(this->sign_result != 1) for HMAC paper version
  //{
 //   sign_resultsFlag = false;
  //}
 //   verification_resultsFlag = sign_resultsFlag;
  //std::cout << "here";

  //  if (verification_resultsFlag = true)
   // {
   //   this->verified_symbols.push_back(h_appendedSymbol);
  //  }
   // return verification_resultsFlag;
  
                                                            //}
                                                            // bool MAC_resultsFlag = true;
                                                            // return MAC_resultsFlag;
//};
