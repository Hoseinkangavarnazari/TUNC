from pyfinite import ffield

F = ffield.FField(8)

a = 7
b = 100


c = F.Multiply(a, b)
# F.Add(a,b)

print(c)
F.ShowPolynomial(c)

file_path = "result.txt"
with open(file_path, "w") as file:
    for i in range(0, 256):
        for j in range(0, 256):
            print(F.Multiply(i, j))
            file.write(str(F.Multiply(i, j)) + "-")
        file.write("\n")

file_path = "addition.txt"
with open(file_path, "w") as file:
    for i in range(0, 256):
        for j in range(0, 256):
            print(F.Add(i, j))
            file.write(str(F.Add(i, j)) + "-")
        file.write("\n")
