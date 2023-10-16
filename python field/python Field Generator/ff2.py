from pyfinite import ffield

F = ffield.FField(2)

file_path = "multiplication_ff_2.txt"
with open(file_path, "w") as file:
    for i in range(0, 4):
        for j in range(0, 4):
            print(F.Multiply(i, j))
            file.write(str(F.Multiply(i, j)) + "-")
        file.write("\n")

        akif



       