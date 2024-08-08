//--------------------------ORIGINAL KDC---------------------------------------//
// std::vector<std::vector<uint8_t>> keyDistributor(int _KeysetSize, std::vector<std::vector<uint8_t>> KeyPool)
// {
//  // std::cout << "here";

//   std::vector<int> chosenSet;
//   std::vector<std::vector<uint8_t>> assignedKeyset(_KeysetSize, std::vector<uint8_t>(KeyPool[0].size(),0));

//   std::vector<int> chosenNumbers(_KeysetSize, 0);
//   // Create a discrete distribution based on the probabilities
//   std::uniform_int_distribution<int> distribution(0, KeyPool.size() - 1);
//   // Sample an index based on probabilities
//   std::set<int> uniqueValues;

//   for (int i = 0; i < _KeysetSize; i++)
//   {
//     int generatedValue = 0;

//     do
//     {
//       generatedValue = distribution(generator);
//     } while (!uniqueValues.insert(generatedValue).second); // Continue generating if the value is not unique

//     chosenNumbers[i] = generatedValue;
//   };
//   for (int i = 0; i < _KeysetSize; i++)
//   {

//     assignedKeyset[i] = KeyPool[chosenNumbers[i]];
//   };
//   return assignedKeyset;
// };
//------------------------------------------------------------------------------------------------------------------------//
// int mainOLD()           ///////////MAIN START/////////////////
// {

//   // // std::vector<std::vector<uint8_t>> test_mac;
//   // hpacket hp(hcodedSymbol, publickeyset, publickeyset.size());
//   // // test_mac = macCalculator(hcodedSymbol, publickeyset, 5);
//   // FieldSize fieldSize = BinaryC;
//   // int dataSizeInBytes = generationSize * symbolSize;
//   // std::vector<uint8_t> sampleData = randomDataGenerator(dataSizeInBytes, fieldSize);
//   // std::vector<std::vector<uint8_t>> test_mac = hp.macCalculator(hcodedSymbol, publickeyset, publickeyset.size());
//   // rlnc_encoder encoder(generationSize, symbolSize, fieldSize);
//   // rlnc_decoder decoder(generationSize, symbolSize, fieldSize);
//   // bool res = encoder.setSymbol(sampleData);
//   // // while
//   // std::vector<packet> allPackets;
//   // int packetCount = 0;
//   // while (packetCount != generationSize)
//   // {
//   //   // create a coded packet
//   //   // First, generate a random coefficient
//   //   std::vector<uint8_t> tempCoeff;
//   //   encoder.randomCoeffGenerator(tempCoeff);
//   //   // Second, encode using the coeff
//   //   std::vector<uint8_t> codedSybmols = encoder.encode_rlnc(tempCoeff);
//   //   // Third, create a packet object
//   //   // packet p(tempCoeff, codedSybmols,packetCount);

//   //   // give the packet to the decoder
//   //   // allPackets.push_back(p);
//   //   // decoder.consumeCodedPacket(p);
//   //   // packetCount++;
//   // }
//   // decoder.decode();

//   // std::vector<std::vector<uint8_t>> key1 = {{2, 3, 1, 4, 2}, {5, 8, 1, 19, 35}, {75, 15, 7, 9, 58}, {175, 156, 47, 91, 67}, {64, 23, 43, 56, 198}, {92, 37, 15, 42, 26}, {53, 82, 11, 91, 53}, {57, 51, 8, 6, 85}, {17, 106, 74, 19, 76}, {60, 25, 45, 55, 178}};

//   // std::vector<uint8_t> private_key1 = {2, 4, 13, 111, 68};
//   //  std::vector<uint8_t> private_key2 = {1, 7, 123, 8};
//   //  std::vector<uint8_t> private_key3 = {55, 58, 71};
//   // std::vector<uint8_t> private_key = {5, 4, 3, 2, 1, 67, 89, 12, 23, 33, 234};

//   std::vector<std::vector<uint8_t>> MACs;
//   std::vector<std::vector<uint8_t>> verified_symbols;

//   // uint8_t currentsign;
//   //  uint8_t c_sign;
//   // uint8_t sign;
//   //  Create a directed graph with 4 nodes
//   //   Graph myGraph(4);
//   //  std::vector<uint8_t> cs2 = {7, 4, 2};
//   //  std::vector<std::vector<uint8_t>> key2 = {{2, 3, 1, 4},{2, 3, 1, 3},{1,2,3,4}};

//   //  hpacket p2(cs2, MACs, key2, private_key2, 3);

//   // std::vector<uint8_t> cs3 = {8, 4};
//   // std::vector<std::vector<uint8_t>> key3 = {{2, 3, 1},{2, 1, 7}};
//   // hpacket p3(cs3, MACs, key3, private_key3, 2);

//   // std::cout << "test";

//   int counter = 0;
//   int examinationsNumber = 1 * 1000;
//   std::vector<double> sum;
//   std::vector<std::chrono::duration<double>> timer;
//   std::vector<std::chrono::duration<double>> timer_simple_verifier;
//   std::vector<std::chrono::duration<double>> timer_tree_verifier;
//   std::vector<std::chrono::duration<double>> timer_ar_verifier;
//   std::vector<std::chrono::duration<double>> mean_timer_simple_verifier;
//   std::vector<std::chrono::duration<double>> mean_timer_tree_verifier;
//   std::vector<std::chrono::duration<double>> mean_timer_ar_verifier;
//   std::vector<std::chrono::duration<double>> timer_summation;
//   std::vector<std::chrono::duration<double>> timer_verification;
//   std::vector<std::chrono::duration<double>> timer_multiplication;
//   std::vector<std::chrono::duration<double>> mean_timer_multiplication;
//   std::vector<std::chrono::duration<double>> timerCombiner;
//   std::vector<std::chrono::duration<double>> mean_timer_summation;
//   int minPacketSize = 100;
//   int maxPacketSize = 501;
//   int packetStep = 10;
//   int minMACSize = 2;
//   int maxMACSize = 23;
//   int MACStep = 5;
//   int NumberOfLayers = 8;

//   int NUmberOfLeaves = 2;
//   double generationSize_double = std::pow(NUmberOfLeaves, NumberOfLayers - 1);
//   int generationSize = static_cast<int>(generationSize_double);
//   int NumberOfPOllutedPackets = 3;
//   // int packetSize = 5;
//   int MACNumber = 3;
//   int pollutionNumber = 2;
//   std::vector<int> simple_counter(examinationsNumber, 0);
//   std::vector<int> tree_counter(examinationsNumber, 0);
//   std::vector<int> ar_tree_counter(examinationsNumber, 0);

//   // change the file name based on the given steup

//   // std::string filename = "./Results/Packet" + std::to_string(minPacketSize) + "-" + std::to_string(maxPacketSize);
//   // filename += "MAC" + std::to_string(minMACSize) + "-" + std::to_string(maxMACSize);
//   // filename += "ExNum:" + std::to_string(examinationsNumber);
//   // filename += ".txt";
//   int cnt = 0;
//   for (int packetSize = minPacketSize; packetSize < maxPacketSize; packetSize += packetStep)
//   {
//     // for (int pollutionNumber = 1; pollutionNumber < 5; pollutionNumber++)
//     //{
//     //  for (int MACNumber = minMACSize; MACNumber < maxMACSize; MACNumber += MACStep)
//     //  {

//     std::string filename = "./Results/Packet" + std::to_string(packetSize);
//     filename += "Generation Size" + std::to_string(generationSize);
//     filename += "Number of Polluted Packets" + std::to_string(pollutionNumber);
//     filename += "ExNum:" + std::to_string(examinationsNumber);
//     filename += ".txt";

//     std::ofstream outputFile(filename, std::ios::app);

//     if (!outputFile.is_open())
//     {
//       std::cerr << "Error opening the file!" << std::endl;
//       return 1;
//     };
//     // generate the keys for MACs based on the packet size
//     std::vector<std::vector<uint8_t>> key1(MACNumber, std::vector<uint8_t>(packetSize + 1, 0));
//     for (int i = 0; i < MACNumber; i++)
//     {
//       std::vector<uint8_t> newKey = generateRandomVector(packetSize + 1);
//       key1[i] = newKey;
//     };
//     // generate the keys for sign based on the packet size
//     // std::vector<uint8_t> private_key;
//     // for (int i = 0; i < MACNumber; i++)
//     //{
//     std::vector<uint8_t> private_key = generateRandomVector(MACNumber + 1);
//     // private_key.push_back(newPrivateKey);
//     //};

//     // int total_time=0;
//     // timer.clear();
//     // timer_multiplication.clear();
//     // timer_summation.clear();
//     // timerCombiner.clear();
//     for (int i = 0; i < examinationsNumber; i++)
//     {
//       if (i % 100 == 0)
//       {
//         std::cout << "i :" << i << std::endl;
//       }
//       //      std::vector<int> probabilities = generateRandomARvector(generationSize);
//       std::vector<int> probabilities = initializeARvector(generationSize);
//       //        std::cout << "here";

//       std::vector<std::vector<uint8_t>> receivedPackets(generationSize, std::vector<uint8_t>(packetSize, 0));
//       // std::cout << "here";

//       std::vector<uint8_t> coefficientVector = generateRandomVector(generationSize);
//       // create an hpacket with the random data
//       for (int j = 0; j < generationSize; j++)
//       {

//         std::vector<uint8_t> cs1 = generateRandomVector(packetSize);
//         receivedPackets[j] = cs1;
//       };

//       hpacket p1(receivedPackets, MACs, key1, private_key, MACNumber, coefficientVector);
//       // TreeGenerator x1(receivedPackets);
//       int a = packetSize + MACNumber + 1;
//       std::vector<std::vector<uint8_t>> verifierSymbols(generationSize, std::vector<uint8_t>(a, 0));

//       // start the timer
//       // auto start = std::chrono::high_resolution_clock::now();

//       // check the integrity
//       p1.macCalculator(); // fixed
//       p1.signCalculator();
//       // p1.packetAppender(receivedPackets);                   // fixed
//       verifierSymbols = p1.packetAppender(receivedPackets); // fixed
//       std::vector<int> pIv = pollutionIndexselector(generationSize, pollutionNumber, probabilities);
//       //    std::cout << "here";

//       //    p1.macVerifier(verifierSymbols[0]);      //fixed
//       // std::cout << "here";

//       // p1.signVerifier(verifierSymbols[0]);     //fixed
//       // std::cout << "here";
//       // std::cout << "here";
//       ///////////////////////////////////////  POLLUTION GENERATOR ////////////////////////////
//       std::vector<std::vector<uint8_t>> pollutedVerifierSymbols = p1.pollutionGeneration(verifierSymbols, pIv);
//       std::vector<std::vector<std::vector<uint8_t>>> verificationTree = p1.treeGenerator(pollutedVerifierSymbols, NumberOfLayers, NUmberOfLeaves, verifierSymbols[0].size()); /// fixed

//       ///////////////////////////////////////  TREE GENERATOR & VERIFIER and TIME MEASUREMENT ///////////////////////////
//       // activate again  auto start_tree_verifier = std::chrono::high_resolution_clock::now();
//       // std::cout << "here";
//       // std::cout << "The round starts here";
//       tree_counter[i] = p1.treeVerifier(verificationTree, NumberOfLayers, NUmberOfLeaves);
//       // activate again  auto end_tree_verifier = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_tree_verifier = std::chrono::duration_cast<std::chrono::microseconds>(end_tree_verifier- start_tree_verifier);
//       // activate again  timer_tree_verifier.push_back(duration_tree_verifier);
//       //  std::cout << "TreeVerifier done";
//       ///////////////////////////////////////  SIMPLE VERIFIER and TIME MEASUREMENT ///////////////////////////
//       // activate again auto start_simple_verifier = std::chrono::high_resolution_clock::now();
//       simple_counter[i] = p1.simpleVerifier(verifierSymbols); // fixed
//       // activate again  auto end_simple_verifier = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_simple_verifier = std::chrono::duration_cast<std::chrono::microseconds>(end_simple_verifier- start_simple_verifier);
//       // activate again  timer_simple_verifier.push_back(duration_simple_verifier);
//       ///////////////////////////////////////  TREE GENERATOR & AR VERIFIER and TIME MEASUREMENT ///////////////////////////
//       // activate again auto start_ar_verifier = std::chrono::high_resolution_clock::now();
//       //  std::vector<std::vector<std::vector<uint8_t>>> verificationTree_ar = p1.treeGenerator(pollutedVerifierSymbols, NumberOfLayers, NUmberOfLeaves, verifierSymbols[0].size()); /// fixed
//       // std::cout << "simpleverifier done";
//       ar_tree_counter[i] = p1.arTreeVerifier(verificationTree, probabilities, NumberOfLayers); // AR based tree algorithm done !!
//       // activate again  auto end_ar_verifier = std::chrono::high_resolution_clock::now();
//       // activate again auto duration_ar_verifier = std::chrono::duration_cast<std::chrono::microseconds>(end_ar_verifier- start_ar_verifier);
//       // activate again  timer_ar_verifier.push_back(duration_ar_verifier);
//       //  std::cout << "ARTreeVerifier done";

//       std::vector<uint8_t> rdnm1 = generateRandomVector(packetSize);
//       std::vector<uint8_t> rdnm2 = generateRandomVector(packetSize);
//       uint8_t rndmnmbr = 5;
//       ///////////////////////////// timer for single addition //////////////////////////////////
//       // activate again   auto start_combiner = std::chrono::high_resolution_clock::now();
//       p1.randomCombiner(rdnm1, rdnm2);
//       // activate again  auto end_combiner = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_combiner = std::chrono::duration_cast<std::chrono::microseconds>(end_combiner - start_combiner);
//       // activate again  timer_summation.push_back(duration_combiner);
//       ///////////////////////////// timer for single verification //////////////////////////////////
//       // activate again  auto start_verifier = std::chrono::high_resolution_clock::now();
//       p1.macVerifier(verifierSymbols[0]);
//       p1.signVerifier(verifierSymbols[0]);
//       // activate again  auto end_verifier = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_verifier = std::chrono::duration_cast<std::chrono::microseconds>(end_verifier - start_verifier);
//       // activate again  timer_verification.push_back(duration_verifier);
//       ///////////////////////////// timer for single multiplication //////////////////////////////////
//       // activate again  auto start_multiplier = std::chrono::high_resolution_clock::now();
//       p1.randomMultiplier(rndmnmbr, rdnm1);
//       // activate again  auto end_multiplier = std::chrono::high_resolution_clock::now();
//       // activate again  auto duration_multiplier = std::chrono::duration_cast<std::chrono::microseconds>(end_multiplier - start_multiplier);
//       // activate again timer_multiplication.push_back(duration_multiplier);
//     };
//     // stop the timer
//     // auto end = std::chrono::high_resolution_clock::now();

//     // print the result and put it in the file
//     //   outputFile << "PacketSize:" << packetSize << "-"
//     //            << "MACSize:" << MACNumber << "-Result:" << sum_size / examinationsNumber << std::endl;
//     // outputFile << "PacketSize:" << packetSize << "-"
//     //         << "MACSize:" << MACNumber << "-ResultCombiner:" << sum_sizeCombiner / examinationsNumber << std::endl;
//     // outputFile.flush();
//     //  print the result and put it in the file
//     ////////////////////////// TAKING MEAN VALUES FOR MEASUREMENTS  ////////////////////////////////////
//     //////// combination
//     // activate again   auto totalDuration_sum = std::accumulate(timer_summation.begin(), timer_summation.end(), std::chrono::duration<double>(0));
//     // activate again  auto sum_mean =totalDuration_sum/ examinationsNumber;
//     //////// verification
//     // activate again  auto totalDuration_verification = std::accumulate(timer_verification.begin(), timer_verification.end(), std::chrono::duration<double>(0));
//     // activate again  auto verification_mean =totalDuration_verification/ examinationsNumber;
//     //   mean_timer_summation[cnt]= sum_mean;
//     //////// multiplication
//     // activate again  auto totalDuration_multiply=std::accumulate(timer_multiplication.begin(), timer_multiplication.end(), std::chrono::duration<double>(0)) ;
//     // activate again  auto multip_mean = totalDuration_multiply/examinationsNumber;
//     ////////  simple verifier
//     // activate again  auto totalDuration_simple = std::accumulate(timer_simple_verifier.begin(), timer_simple_verifier.end(), std::chrono::duration<double>(0));
//     // activate again  auto simple_verifier_mean =totalDuration_simple/ examinationsNumber;
//     ////////  tree verifier
//     // activate again  auto totalDuration_tree = std::accumulate(timer_tree_verifier.begin(), timer_tree_verifier.end(), std::chrono::duration<double>(0));
//     // activate again  auto tree_verifier_mean =totalDuration_tree/ examinationsNumber;
//     ////////  ar verifier
//     // activate again  auto totalDuration_ar = std::accumulate(timer_ar_verifier.begin(), timer_ar_verifier.end(), std::chrono::duration<double>(0));
//     // activate again  auto ar_verifier_mean =totalDuration_ar/ examinationsNumber;

//     // mean_timer_multiplication[cnt] = multip_mean;
//     int avg_simple = std::accumulate(simple_counter.begin(), simple_counter.end(), 0) / examinationsNumber;
//     int avg_tree = std::accumulate(tree_counter.begin(), tree_counter.end(), 0) / examinationsNumber;
//     int avg_ar_tree = std::accumulate(ar_tree_counter.begin(), ar_tree_counter.end(), 0) / examinationsNumber;
//     cnt++;

//     std::cout << "here";

//     outputFile << "PacketSize:" << packetSize << "-"
//                << "GenerationSize:" << generationSize << "Pollution Number" << pollutionNumber << "-SImple Ver Check Number:" << avg_simple << std::endl;

//     outputFile << "PacketSize:" << packetSize << "-"
//                << "GenerationSize:" << generationSize << "Pollution Number" << pollutionNumber << "-Tree VerResult:" << avg_tree << std::endl;
//     outputFile << "PacketSize:" << packetSize << "-"
//                << "GenerationSize:" << generationSize << "Pollution Number" << pollutionNumber << "-AR Tree Ver Result:" << avg_ar_tree << std::endl;
//     outputFile.flush();
//     //}
//     outputFile << std::endl;
//     outputFile.flush();
//   };
//   // Set inputs for Node 0, Node 1, and Node 2 with different types of data
//   //  NodeInput input0 = {{1, 2, 3}, {{4, 5}, {6, 7}}, 8};
//   //  NodeInput input1 = {{9, 10}, {{11, 12}, {13, 14}}, 15};
//   //   NodeInput input2 = {{16, 17, 18}, {{19, 20}, {21, 22}}, 23};

//   //   myGraph.setInput(0, input0);
//   //  myGraph.setInput(1, input1);
//   //   myGraph.setInput(2, input2);

//   // Add directed edges between nodes
//   //  myGraph.addDirectedEdge(0, 1);
//   //   myGraph.addDirectedEdge(0, 2);

//   // std::cout << "here";

//   //      std::cout << "here";

//   return 0;
// };          /////////   MAIN END ///////

// .............................................................................................
