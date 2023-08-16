from pyfinite import ffield

F = ffield.FField(8)

for i in range(1,256):
    print("i = ",i, " = ",F.Inverse(i))
