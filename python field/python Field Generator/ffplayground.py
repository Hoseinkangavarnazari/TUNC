from pyfinite import ffield

F = ffield.FField(8)


print(F.Add(164,9))


#for i in range(0, 255):
#    for j in range(0, 255):
#        res = F.Add(i, j)
#        if res == 0:
#           print(i, " is inverse of ", j)


# for i in range(1,256):
#     print("i = ",i, " = ", F.Divide(10,i))


# a = [110, 120, 30, 43 ]


# for i in range(0,4):
#     print("i = ",i, " = ", F.Divide(a[i],255))


#Take it back
# // stop-start
#       auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
#       timer.push_back(duration);

#       // start the timer for combiner
#       auto startCombiner = std::chrono::high_resolution_clock::now();
#       // combine packets
#       //   p1.packetCombiner();
#       // stop the timer
#       auto endCombiner = std::chrono::high_resolution_clock::now();
#       // stop-start
#       auto durationCombiner = std::chrono::duration_cast<std::chrono::nanoseconds>(endCombiner - startCombiner);
#       timerCombiner.push_back(durationCombiner);
#     };
#   //  std::cout << "here";
#     double sum_size = 0;
#     for (const auto &entry : timer)
#     {
#       sum_size += entry.count(); // Adding each duration to the sum in milliseconds
#     };
#     sum.push_back(sum_size / examinationsNumber);
#     //
#     double sum_sizeCombiner = 0;
#     for (const auto &entry : timerCombiner)
#     {
#       sum_sizeCombiner += entry.count(); // Adding each duration to the sum in milliseconds
#     };
#     sum.push_back(sum_sizeCombiner / examinationsNumber);