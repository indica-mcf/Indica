from indica.readers import ADASReader

reader = ADASReader()
scd_W = reader.get_adf11("scd", "W", "1989")
plt_W = reader.get_adf11("plt", "W", "1989")
scd_H = reader.get_adf11("scd", "H", "1996")
plt_H = reader.get_adf11("plt", "H", "1996")

print("Effective Ionisation Coefficient of Tungston")
print("--------------------------------------------")
print(scd_W)
print()

print("Line Emission Power for Tungston")
print("--------------------------------")
print(plt_W)
print()

print("Effective Ionisation Coefficient of Hydrogen")
print("--------------------------------------------")
print(scd_H)
print()

print("Line Emission Power for Hydrogen")
print("--------------------------------")
print(plt_H)
print()
