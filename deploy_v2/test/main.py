import subprocess

cmd = ['sh', 'run.sh']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

for line in iter(p.stdout.readline, b''):
    print(line.decode(), end='')
# while True:
#     line = p.stdout.readline()
#     if not line:
#         break
#     print(line.rstrip())
