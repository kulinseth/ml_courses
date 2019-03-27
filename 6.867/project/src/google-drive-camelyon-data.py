import subprocess

# Gdrive is a tool to access the Files on Google Drive through command line.
# The following is a query to generate all the FileId names belonging to the
# Camelyon dataset. These FileIds were used to get the file thorugh a Python
# script. This was essential as it was not possible to do using a web-interface
a = subprocess.check_output(["./gdrive", "list", "--query", "name contains 'patient'",
                    "--absolute", "--max",  "10000", "--no-header"])
b = a.decode("utf-8")
b = b.split('\n')
for i in b:
    c = i.split(" ")
    c = list(filter(None, c))
    if c:
        name = c[1].split('/')
        print (c[0], name[-1])

