from pyfinite import ffield

F = ffield.FField(8)

# for i in range(1,256):
#     print("i = ",i, " = ", F.Divide(10,i))


a = [110, 120, 30, 43 ]

    
for i in range(0,4):
    print("i = ",i, " = ", F.Divide(a[i],255))
   