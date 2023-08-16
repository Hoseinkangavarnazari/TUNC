from pyfinite import ffield

F = ffield.FField(8)

# for i in range(1,256):
#     print("i = ",i, " = ", F.Divide(10,i))
    
for i in range(1,256):
    print("i = ",i, " = ", F.Subtract(8,i))
   