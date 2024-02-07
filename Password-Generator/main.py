import string
import random

# variable holds string format of ascii letters & digits in a list
characters = list(string.ascii_letters + string.digits + "!@#$%^&*")

def password_generator():
    pass_len = int(input("Enter the length of the password: "))

    password = []

    # Shuffling the string values in the characters variable
    random.shuffle = characters
    for y in range(pass_len):

        # Adding random choice of values to password list
        password.append(random.choice(characters))

    # Shuffling the string values in the password list
    random.shuffle = password

    # Concatenation of values into single string
    # without any separation
    password="".join(password)

    print(password)

option = input("Do you want to generate a password (Yes/No)?")
option = option.lower()
if option == "yes":
    password_generator()
elif option == "no":
    print("Program terminated")
    quit()
else:
    print("Invalid input!!! Please enter Yes/No")
    quit()
    