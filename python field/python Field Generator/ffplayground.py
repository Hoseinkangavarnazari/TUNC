from pyfinite import ffield

F = ffield.FField(8)


print(F.Multiply(28,3))


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
