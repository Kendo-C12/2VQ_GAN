s = input("Enter the string: ")

# Reference sequence A-Z
seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Flag to track if string is sorted like repeated ABC...
is_sorted = True

for i, c in enumerate(s):
    # The expected character at this position
    expected = seq[i % 26]  # modulo 26 to wrap around
    if c != expected:
        is_sorted = False
        print(f"Mismatch at position {i}: got '{c}', expected '{expected}'")
        break

if is_sorted:
    print("The string is a correctly sorted repeating sequence A-Z.")
else:
    print("The string is NOT a correctly sorted repeating sequence A-Z.")
