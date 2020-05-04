import subprocess as cmd

cp = cmd.run("git add .", check=True, shell=True)
print(cp)

# response = input("Do you want to use the default message for this commit?([y]/n)\n")
# message = "auto-update"

# if response.startswith('n'):
#     message = input("Type commit message\n")

message = 'auto-update'

cp = cmd.run(f"git commit -m '{message}'", check=True, shell=True)
cp = cmd.run("git push covid-19-dashboard master", check=True, shell=True)