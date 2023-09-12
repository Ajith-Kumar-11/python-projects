import string
import random

characters = list(string.ascii_letters + string.digits + "!@#$%^&*")

def password_generator():
    pass_len = int(input("Enter the length of the password: "))

    password = []
    random.shuffle = characters
    for y in range(pass_len):
        password.append(random.choice(characters))

    random.shuffle = password
    password="".join(password)

    print(password)

option = input("Do you want to generate a password (Yes/No)?  ")
if option == "Yes":
    password_generator()
elif option == "No":
    print("Program terminated")
    quit()
else:
    print("Invalid input!!! Please enter Yes/No")
    quit()
