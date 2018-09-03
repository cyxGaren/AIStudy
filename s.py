import socket
import time
sa = socket.socket()
sa.bind((socket.gethostname(),19800))
print('aaa')
sa.listen(1024)
while(True):
	sa.accept()
	time.sleep(10000)
print('m')
