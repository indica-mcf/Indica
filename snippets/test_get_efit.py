import getpass

from indica.readers import PPFReader

reader = PPFReader(90279, 45.0, 50.0)
if reader.requires_authentication:
    user = input("JET username: ")
    password = getpass.getpass("JET password: ")
    assert reader.authenticate(user, password)

data = reader.get("jetppf", "efit", 0)
print("f value")
print("=======\n")
print(data["f"])
print("\n\n")
print("psi")
print("===\n")
print(data["psi"])
print("\n\n")
print("Rmag")
print("====\n")
print(data["rmag"])
print("\n\n")
print("Rbnd")
print("====\n")
print(data["rbnd"])
