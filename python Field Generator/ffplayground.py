from pyfinite import ffield

F = ffield.FField(8)

# for i in range(1,256):
#     print("i = ",i, " = ", F.Divide(10,i))


a = [11, 12, 3, 43 ]
b = [51, 61, 23, 8 ]
    
for i in range(0,4):
    print("i = ",i, " = ", F.Subtract(a[i],b[i]))
   