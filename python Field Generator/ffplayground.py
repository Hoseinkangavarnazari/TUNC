from pyfinite import ffield

F = ffield.FField(8)

i = 11
j = 15
print(i, j, F.Multiply(i, j))
